# README

## Project Description

This package contains the numerical codes and example files used to reproduce the results presented in the manuscript:

> **A Dimensionless Numerical Framework for Modeling Wave-Induced Fluid Flow in Heterogeneous Porous Media**

In addition, simplified CPU-based versions are provided for small test cases, including one-dimensional homogeneous and layered media, to facilitate code verification and demonstration of the numerical implementation.

The numerical solver is based on **Biot's elastodynamic theory of poroelasticity**. The framework allows the computation of velocity dispersion and attenuation associated with wave-induced fluid flow (WIFF) using a dimensionless scaling approach.

---

# File List

| File | Description |
|------|-------------|
| `main.m` | Main MATLAB script. |
| `init_data3D_v25.m` | Pre-processing script for domain decomposition for MPI. |
| `MPI_Biot_ZY.cu` | CUDA-MPI solver for 3-D poroelastic wave propagation. |
| `compute_Vp_Q_from_Vx.m` | Post-processing routine for computing velocity and attenuation. |
| `geocomp_unil_mpi3D_v4.m` | MPI configuration module. |

---

# Software Requirements

The following software is required:

- MATLAB R2022a or later
- CUDA Toolkit 12.0 or later
- OpenMPI 4.1.8 or later
- GCC 8.5.0 or later
- Linux operating system

---

# Running the Code

## Step 1. Pre-processing

### 1. Model Generation

Construct a numerical model represented by a label matrix.

Each grid cell is assigned an integer label corresponding to a specific material or geometric feature.

For example,

- `0` represents fractures.
- `1` represents the background medium.

Additional labels can be defined for other materials or fluid-saturation regions as needed.

---

### 2. Build Material Property Matrices and Partition the Domain for MPI

Before running the simulation, specify the number of GPUs to be used.

The computational domain is then automatically partitioned into subdomains according to the GPU configuration for MPI-based parallel execution in **main.m**.

```matlab
%% ========================================================================
%  3. MPI-GPU parameters
% ========================================================================

dims_x = 4;
dims_y = 1;
dims_z = 1;

nx_l = nx / dims_x;
ny_l = ny / dims_y;
nz_l = nz / dims_z;

OVERLENGTH_X = 2 * (dims_x - 1);
OVERLENGTH_Y = 0;
OVERLENGTH_Z = 0;

machineformat = 'n';
PRECIS = 8;

Total_GPU = dims_x * dims_y * dims_z;
GPU_x = dims_x;
GPU_y = dims_y;
GPU_z = dims_z;

NBX = fix(nx_l / 32);
NBY = fix(ny_l / 4);
NBZ = fix(nz_l / 8);
```

In this example, the computational domain is divided into four subdomains along the **x-direction**, resulting in four GPUs being used for the simulation.

After assigning the material properties and applying the dimensionless scaling parameters, the resulting property matrices are partitioned into `n` MPI subdomains (here `n = dims_x`). Each subdomain is subsequently written to a separate binary file for distributed GPU computation.

The partitioning is performed using the following commands in **main.m**:

```matlab
init_data3D_v25(G(ix,iy,iz),      'G',      0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
init_data3D_v25(CM1_11(ix,iy,iz), 'CM1_11', 0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
init_data3D_v25(CM1_12(ix,iy,iz), 'CM1_12', 0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
init_data3D_v25(CM1_22(ix,iy,iz), 'CM1_22', 0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
init_data3D_v25(CM2_11(ix,iy,iz), 'CM2_11', 0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
init_data3D_v25(CM2_12(ix,iy,iz), 'CM2_12', 0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
init_data3D_v25(CM2_22(ix,iy,iz), 'CM2_22', 0, nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);

init_data3D_v25(etaf_k(ix,iy,iz), 'etaf_k', 0, ...
nx_l, ny_l, nz_l, dims_x, dims_y, dims_z, machineformat, PRECIS);
```

More details about MPI implementation can be found in the paper:

> **Resolving Wave Propagation in Anisotropic Poroelastic Media Using Graphical Processing Units (GPUs)**

---

## Step 2. Compilation and Execution

After generating the model and partitioning the domain, compile and launch the CUDA-MPI solver from **main.m**.

```matlab
%% --------------------------------------------------------------------
% Run CUDA-MPI solver
%% --------------------------------------------------------------------

pathToScript = fullfile(pwd, 'runme_volta.sh');

Bash_run = [pathToScript, ' ', ...
int2str(Total_GPU), ' ', ...
int2str(GPU_x), ' ', ...
int2str(GPU_y), ' ', ...
int2str(GPU_z), ' ', ...
int2str(length(pa1)), ' ', ...
int2str(NBX), ' ', ...
int2str(NBY), ' ', ...
int2str(NBZ)];

fprintf('Running command:\n%s\n', Bash_run);

system(Bash_run);
```

This command executes `runme_volta.sh`, which compiles and launches the CUDA-MPI solver using the specified MPI decomposition and GPU configuration.

The compilation command is

```bash
nvcc -arch=sm_70 \
--compiler-bindir mpic++ \
--compiler-options -O3 \
MPI_Biot_ZY.cu \
-DD_x=$2 -DD_y=$3 -DD_z=$4 \
-DNPARS1=$5 \
-DNBX=$6 -DNBY=$7 -DNBZ=$8
```

GPU architectures:

| GPU | Architecture |
|------|--------------|
| V100 | `sm_70` |
| A100 | `sm_80` |
| H100 | `sm_90` |
| RTX 3090 | `sm_86` |

Update `runme_volta.sh` according to your GPU before compilation.

---

## Important Notes

To reduce GPU memory consumption, receiver responses are averaged over the receiver plane during the simulation instead of storing the complete wavefield at every receiver location.

This significantly reduces GPU memory usage and enables larger-scale simulations.

---

## Compatibility with Older GPU Architectures

The code uses double-precision `atomicAdd()`, which is supported only on GPUs with compute capability **6.0 or higher**.

For older architectures (`sm_52`, `sm_50`, etc.), insert the fallback implementation immediately after

```cpp
#include "time.h"
```

in `MPI_Biot_ZY.cu`, and remove the surrounding comment symbols.

```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(
                val + __longlong_as_double(assumed)
            )
        );
    } while (assumed != old);

    return __longlong_as_double(old);
}

#endif
```

---

## Step 3. Simulation

Launch the simulation using MPI.

```bash
run_cmd="-np $nprocs -rf gpu_rankfile_volta \
--mca pml ob1 \
--mca btl self,vader,tcp"

echo $mpirun $run_cmd

$mpirun $run_cmd ./a.out
```

Each MPI process loads its corresponding subdomain and performs the simulation independently on its assigned GPU.

Receiver responses are written to disk for subsequent post-processing.

---

## Step 4. Post-processing

After the simulation completes, compute the dimensionless velocity and attenuation.

```matlab
[Vp_fast, QQ1] = compute_Vp_Q_from_Vx( ...
fname1, fname2, nt, irx1, irx2, dt, dx, freq, false);

Q1(test_n) = abs(QQ1);
Vx_CM(test_n) = Vp_fast;

fprintf('V_nd = %.6e\n', Vp_fast);
fprintf('Q^{-1} = %.6e\n', Q1(test_n));
```

The function `compute_Vp_Q_from_Vx` calculates

- dimensionless phase velocity (`V_nd`)
- inverse quality factor (`Q^{-1}`)

from waveforms recorded at two receiver locations.

---

# Output

The simulation generates:

- Receiver waveforms
  - `x_Vx_rec1_num.res`
  - `x_Vx_rec2_num.res`

  where

  - `x` is the GPU index
  - `num` is the dimensionless parameter index.

- Plane-averaged wavefield snapshots

```
Saved_wavefield_figures/
```

- Velocity-dispersion curves

```
Saved_wavefield_figures/
```

- Attenuation curves

```
Saved_wavefield_figures/
```

- Numerical results

```
Saved_results/
```

---

# Reproducibility

All figures presented in the manuscript can be reproduced using the scripts included in this package together with the parameters described in the manuscript.

The numerical results were generated using **four NVIDIA V100 GPUs**, each with **32 GB memory**.

---

# Contact

**Zhiyu Hou**

University of Lausanne

Email:

- zhiyu.hou@unil.ch
- houzhiyu474@163.com

---

# License

This code is provided solely for academic and non-commercial research purposes.

Copyright © Zhiyu Hou, Yury Alkhimenkov, and Beatriz Quintal.

All rights reserved.
