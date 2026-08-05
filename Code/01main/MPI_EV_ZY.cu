// 3D Elastodynamic Biot's equations in heterogeneous media: ElBIOT3D-MPI , 09 Dec 2025  -  Zhiyu Hou  -  University of Lausanne
// 3D GPU Cuda aware MPI
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"
#define NDIMS  3

//#define USE_SINGLE_PRECISION    /* Comment this line using "//" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT      float
#define PRECIS   4
#else
#define DAT      double
#define PRECIS   8
#endif
#define GPU_ID   3

#define OVERLENGTH_X  0//1
#define OVERLENGTH_Y  0//1
#define OVERLENGTH_Z  0//1
//#define OVERLENGTH_Z  4
#define BOUNDARY_WIDTH_X 16//48  //16 - gives 96percent
#define BOUNDARY_WIDTH_Y 16//16
#define BOUNDARY_WIDTH_Z 16//16

#define zeros(A,nx,ny,nz)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny)*(nz))*sizeof(DAT)); \
                           for(i=0; i < ((nx)*(ny)*(nz)); i++){ A##_h[i]=(DAT)0.0; }              \
                           cudaMalloc(&A##_d      ,((nx)*(ny)*(nz))*sizeof(DAT));                 \
                           cudaMemcpy( A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)        free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny,nz) cudaMemcpy( A##_h,A##_d,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyDeviceToHost);

#define load(A,nx,ny,Aname)  double *A##_d,*A##_h; A##_h = (double*)malloc((nx)*(ny)*sizeof(double));  \
                             FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(double), (nx)*(ny), A##fid); fclose(A##fid); \
                             cudaMalloc(&A##_d,((nx)*(ny))*sizeof(double)); \
                             cudaMemcpy(A##_d,A##_h,((nx)*(ny))*sizeof(double),cudaMemcpyHostToDevice);  
#define  swap(A,B,tmp)         DAT *tmp; tmp = A##_d; A##_d = B##_d; B##_d = tmp;
#define load33(A,nx,ny,nz,Aname)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*(nz)*sizeof(DAT));  \
                             FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(DAT), (nx)*(ny)*(nz), A##fid); fclose(A##fid); \
                             cudaMalloc(&A##_d,((nx)*(ny)*(nz))*sizeof(DAT)); \
                             cudaMemcpy(A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice); 





// --------------------------------------------------------------------- //
// Physics
// Numerics


#define BLOCK_X     32
#define BLOCK_Y     4
#define BLOCK_Z     8
#define GRID_X      NBX // 4*2  //16
#define GRID_Y   NBY// 16*4 //64
#define GRID_Z   NBZ// 64*8 //64


#define DIMS_X   D_x
#define DIMS_Y   D_y
#define DIMS_Z   D_z
const int nx = BLOCK_X * GRID_X - OVERLENGTH_X;
const int ny = BLOCK_Y * GRID_Y - OVERLENGTH_Y;
const int nz = BLOCK_Z * GRID_Z - OVERLENGTH_Z;
//const int nt = 5050;//550; 1050
// Preprocessing
DAT    dx, dy, dz;
size_t Nix, Niy, Niz;
// MPI /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "geocomp_unil_mpi3D_v4.h"
// #include "geocomp_unil_NOmpi3D.h"
// Suboutines /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        
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
void save_info() {
	FILE* fid;
	if (me == 0) { fid = fopen("0_nxyz.inf", "w"); fprintf(fid, "%d %d %d %d", PRECIS, Nix, Niy, Niz);             fclose(fid); }
	if (me == 0) { fid = fopen("0_nxyz_l.inf", "w"); fprintf(fid, "%d %d %d", nx, ny, nz);                        fclose(fid); }
	if (me == 0) { fid = fopen("0_dims.inf", "w"); fprintf(fid, "%d %d %d %d", nprocs, dims[0], dims[1], dims[2]); fclose(fid); }
}
void save_coords() {
	char* fname; FILE* fid; asprintf(&fname, "%d_co.inf", me);
	fid = fopen(fname, "w"); fprintf(fid, "%d %d %d", coords[0], coords[1], coords[2]); fclose(fid); free(fname);
}
void save_array(DAT* A, int nx, int ny, int nz, const char A_name[]) {
	char* fname; FILE* fid; asprintf(&fname, "%d_%s.res", me, A_name);
	fid = fopen(fname, "wb"); fwrite(A, sizeof(DAT), (nx) * (ny) * (nz), fid); fclose(fid); free(fname);
}
#define SaveArray(A,nx,ny,nz,A_name) gather(A,nx,ny,nz); save_array(A##_h,nx,ny,nz,A_name);

void load3(DAT* A_h, DAT* A_d, int nx, int ny, int nz, const char A_name[], const char B_name[], int isave) {
	char* bname; FILE* fid; size_t nb_elems = nx * ny * nz;
	asprintf(&bname, "%d_%d_%s.%s", isave, me, A_name, B_name);
	fid = fopen(bname, "rb"); // Open file
	if (!fid) { fprintf(stderr, "\nUnable to open file %s \n", bname); return; }
	fread(A_h, PRECIS, nb_elems, fid); fclose(fid);
	cudaMemcpy(A_d, A_h, nb_elems * sizeof(DAT), cudaMemcpyHostToDevice);
	if (me == 0) { printf("Read data: %d files %s.%s loaded (size = %dx%dx%d) \n", nprocs, A_name, B_name, nx, ny, nz); } free(bname);
}
#define Load3(A,A_name,B_name,isave)  load3(A##_h, A##_d, size(A,1), size(A,2), size(A,3), A_name, B_name, isave);



#define def_sizes(A,nx,ny,nz)  const int sizes_##A[] = {nx,ny,nz};  
#define      size(A,dim)       (sizes_##A[dim-1])
#define     numel(A)           (size(A,1)*size(A,2)*size(A,3))
int isave = 0;
void save_arrayold(DAT* A, size_t nb_elems, const char A_name[], int isave) {
	char* fname; FILE* fid;
	asprintf(&fname, "%d_%d_%s.res", isave, me, A_name);
	fid = fopen(fname, "wb"); fwrite(A, PRECIS, nb_elems, fid); fclose(fid); free(fname);
}
#define gatherold(A)              cudaMemcpy( A##_h,A##_d,numel(A)*sizeof(DAT),cudaMemcpyDeviceToHost);
#define SaveArrayold(A,A_name)  gatherold(A); save_arrayold(A##_h, numel(A), A_name, isave);
#define   push(A,nx,ny,nz) cudaMemcpy( A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice);
#define x_s    (Lx - Lx/2.0)
#define y_s    (Ly - Ly/2.0)
#define z_s    (Lz - Lz/2.0)
#define pi (DAT)3.14159265358979323846


void pert(const int nt, const DAT f0, const DAT A0, DAT dt, DAT* Src) {
	int itr;
	DAT t0 = (DAT)10.0 / 3e4 / dt;///dt
	for (itr = 0; itr < nt; itr++) {
		//DAT t    = ((DAT)itr)*dt;
		//Src[itr] = A0*exp(-(DAT)0.5*((DAT)4.0*f0)*((DAT)4.0*f0)*(t-t0)*(t-t0));
		Src[itr] = (DAT)0.0 * A0 * ((DAT)1.0 - (DAT)2.0 * (pi * (itr - t0) * dt * f0) * (pi * (itr - t0) * dt * f0)) * exp(-(pi * (itr - t0) * dt * f0) * (pi * (itr - t0) * dt * f0));
	}
}


// Computing physics kernels /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void init(int* coords, DAT* x, DAT* y, DAT* z, DAT* Prf, DAT* sigma_xx, DAT* sigma_yy, DAT* sigma_zz, DAT* Vx, DAT* Vy, DAT* Qxft, DAT* Qyft, DAT* vrhox11hets, const DAT dx, const DAT dy, const DAT dz, const DAT Lx, const DAT Ly, const DAT Lz, const DAT lamx, const DAT lamy, const DAT lamz, const int nx, const int ny, const int nz) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z

	if (iz < nz && iy < ny && ix < nx) { x[ix + iy * nx + iz * nx * ny] = (DAT)(coords[0] * (nx - 2) + ix) * dx - (DAT)0.5 * Lx; }
	if (iz < nz && iy < ny && ix < nx) { y[ix + iy * nx + iz * nx * ny] = (DAT)(coords[1] * (ny - 2) + iy) * dy - (DAT)0.5 * Ly; }
	if (iz < nz && iy < ny && ix < nx) { z[ix + iy * nx + iz * nx * ny] = (DAT)(coords[2] * (nz - 2 * 1) + iz) * dz - (DAT)0.5 * Lz; } //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	if (iz < nz && iy < ny && ix < nx) { Prf[ix + iy * nx + iz * nx * ny] = -(DAT)1.0 * (DAT)10000000000.0 * exp(-(x[ix + iy * nx + iz * nx * ny] * x[ix + iy * nx + iz * nx * ny] / lamx / lamx) - (y[ix + iy * nx + iz * nx * ny] * y[ix + iy * nx + iz * nx * ny] / lamy / lamy) - (z[ix + iy * nx + iz * nx * ny] * z[ix + iy * nx + iz * nx * ny] / lamz / lamz)); }
	//if (iz<nz && iy<ny && ix<nx){ vrhox11hets[ix + iy*nx + iz*nx*ny] = (DAT)0.0006426735218509  +  (DAT)0.0006426735218509*(DAT)0.5*exp(  -(x[ix + iy*nx + iz*nx*ny]*x[ix + iy*nx + iz*nx*ny]/lamx/lamx) -(y[ix + iy*nx + iz*nx*ny]*y[ix + iy*nx + iz*nx*ny]/lamy/lamy) -(z[ix + iy*nx + iz*nx*ny]*z[ix + iy*nx + iz*nx*ny]/lamz/lamz)  ); }

}

__global__ void init2(DAT* vrhox11hets, DAT* vrhox11hetsl, const int nx, const int ny, const int nz, int istep) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
	CommOverlap();
	if (iz < nz && iy < ny && ix>0 && ix < nx) { vrhox11hetsl[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = (vrhox11hets[ix + iy * nx + iz * nx * ny] + (DAT)1.0 * vrhox11hets[(ix - 1) + (iy)*nx + (iz)*nx * ny]) * ((DAT)1.0 / (DAT)2.0); }

}

__global__ void init3(DAT* vrhox11hets, DAT* vrhox11hetsl, const int nx, const int ny, const int nz, int istep) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
	CommOverlap();
	if (iz < nz && iy < ny && iy>0 && ix < nx) { vrhox11hetsl[ix + iy * (nx)+iz * (nx) * (ny + 1)] = (vrhox11hets[ix + iy * nx + iz * nx * ny] + (DAT)1.0 * vrhox11hets[(ix)+(iy - 1) * nx + (iz)*nx * ny]) * ((DAT)1.0 / (DAT)2.0); }

}
__global__ void init4(DAT* vrhox11hets, DAT* vrhox11hetsl, const int nx, const int ny, const int nz, int istep) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
	CommOverlap();
	if (iz < nz && iy < ny && iz>0 && ix < nx) { vrhox11hetsl[ix + iy * (nx)+iz * (nx) * (ny)] = (vrhox11hets[ix + iy * nx + iz * nx * ny] + (DAT)1.0 * vrhox11hets[(ix)+(iy)*nx + (iz - 1) * nx * ny]) * ((DAT)1.0 / (DAT)2.0); }

}
__global__ void source(int* coords, DAT* Pr, DAT* Src, int it, const int nx, const int ny, const int nz, const int nt, DAT* mask) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z

	int ix_g = (coords[0] * (nx - 2) + ix);

	if (ix_g == 0) { 
		//Pr[ix + iy * nx + iz * nx * ny] = (Pr[ix + iy * nx + iz * nx * ny]+ Src[it] )*mask[ix + iy * nx + iz * nx * ny]; 
	Pr[ix + iy * nx + iz * nx * ny] = (Pr[ix + iy * nx + iz * nx * ny] + Src[it]) ;
	//Pr[ix + iy * nx + iz * nx * ny] = (Pr[ix + iy * nx + iz * nx * ny] + Src[it] * (1-mask[ix + iy * nx + iz * nx * ny]));
	
	}
}


__global__ void reciever(int* coords, DAT* Vxs, DAT* Vxf, int irx1, int irx2, DAT* Vx_rec1, DAT* Vx_rec2, int it, const int nx, const int ny, const int nz, const int nt) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z


	int ix_g = (coords[0] * (nx - 2) + ix);

	if (ix_g == irx1 - 1) {
		//Vx_rec1[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];

		atomicAdd(&Vx_rec1[it], Vxs[ix + iy * (nx + 1) + iz * (nx + 1) * ny]);
	}
		if (ix_g == irx2 - 1) {
		//Vx_rec2[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
		atomicAdd(&Vx_rec2[it], Vxf[ix + iy * (nx + 1) + iz * (nx + 1) * ny]);
	}

}
__global__ void reciever_src(int* coords, DAT* Vxs, DAT* Vxf, int irx1, int irx2, DAT* Vx_rec1, DAT* Vx_rec2, int it, const int nx, const int ny, const int nz, const int nt) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z


	int ix_g = (coords[0] * (nx - 2) + ix);

	if (ix_g == 0) {
		//Vx_rec1[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];

		atomicAdd(&Vx_rec1[it], Vxs[ix + iy * (nx + 0) + iz * (nx + 0) * ny]);
	}
		if (ix_g == 0) {
		//Vx_rec2[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
		atomicAdd(&Vx_rec2[it], Vxf[ix + iy * (nx + 0) + iz * (nx + 0) * ny]);
	}


}
__global__ void reciever11(int* coords, DAT* Vxs, DAT* Vxf,int irx1, int irx2, DAT* Vx_rec1, DAT* Vx_rec2, int it, const int nx, const int ny, const int nz, const int nt) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z


	int ix_g = (coords[0] * (nx - 2) + ix);
	if (ix_g == irx1 - 1) {
		//Vx_rec1[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];

		atomicAdd(&Vx_rec1[it], Vxs[ix + iy * (nx + 1) + iz * (nx + 1) * ny]);
	}
		if (ix_g == irx2 - 1) {
		//Vx_rec2[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
		atomicAdd(&Vx_rec2[it], Vxf[ix + iy * (nx + 1) + iz * (nx + 1) * ny]);
	}


}


__global__ void reciever22(int* coords, DAT* Vxs, DAT* Vxf,int irx1, int irx2, DAT* Vx_rec1, DAT* Vx_rec2, int it, const int nx, const int ny, const int nz, const int nt) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z


	int ix_g = (coords[0] * (nx - 2) + ix);
	if (ix_g == irx1 - 1) {
		//Vx_rec1[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];

		atomicAdd(&Vx_rec1[it], Vxs[ix + iy * (nx + 1) + iz * (nx + 1) * ny]);
	}
		if (ix_g == irx2 - 1) {
		//Vx_rec2[it + iy * nt + iz * nt * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
		atomicAdd(&Vx_rec2[it], Vxf[ix + iy * (nx + 1) + iz * (nx + 1) * ny]);
	}


}

__global__ void compute_P(DAT* mask, DAT* Vx, DAT* Vy, DAT* Vz, DAT* P,DAT* Ps,DAT* Pf, DAT* tau_xx, DAT* tau_yy, DAT* tau_zz, DAT* tau_xy, DAT* tau_xz, DAT* tau_yz, const DAT dt, DAT* K, DAT* G, DAT* eta, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const int nt) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
#define diffVx_x  ( (Vx[(ix+1) + iy     *(nx+1)  + iz     *(nx+1)  * ny   ]-Vx[ix  + iy *(nx+1)  + iz *(nx+1)  * ny    ])/dx)
#define diffVy_y  ( (Vy[ ix    +(iy+1)  * nx     + iz     * nx     *(ny+1)]-Vy[ix  + iy * nx     + iz * nx     *(ny+1) ])/dy)
#define diffVz_z  ( (Vz[ ix    + iy     * nx     +(iz+1)  * nx     * ny   ]-Vz[ix  + iy * nx     + iz * nx     * ny    ])/dz)
#define diffVx_y  ( (Vx[ ix    + iy     *(nx+1)  + iz     *(nx+1)  * ny   ]-Vx[ix  +(iy-1)*(nx+1)+ iz *(nx+1)  * ny    ])/dy)
#define diffVy_x  ( (Vy[ ix    + iy     * nx     + iz     * nx     *(ny+1)]-Vy[ix-1+ iy * nx     + iz * nx     *(ny+1) ])/dx)
#define diffVx_z  ( (Vx[ ix    + iy     *(nx+1)  + iz     *(nx+1)  * ny   ]-Vx[ix  + iy *(nx+1)  + (iz-1) *(nx+1)  * ny])/dz)
#define diffVz_x  ( (Vz[ ix    + iy     * nx     + iz     * nx     * ny   ]-Vz[ix-1+ iy * nx     + iz * nx     * ny    ])/dx)
#define diffVy_z  ( (Vy[ ix    + iy     * nx     + iz     * nx     *(ny+1)]-Vy[ix  + iy * nx     + (iz-1) * nx     *(ny+1) ])/dz)
#define diffVz_y  ( (Vz[ ix    + iy     * nx     + iz     * nx     * ny   ]-Vz[ix  + (iy-1) * nx     + iz * nx     * ny    ])/dy)
	if (iz < nz && iy < ny && ix < nx) {
		DAT diffV = diffVx_x + diffVy_y + diffVz_z;
		DAT diffVxx = diffVx_x;
		DAT diffVyy = diffVy_y;
		DAT diffVzz = diffVz_z;
		P[ix + iy * nx + iz * nx * ny] = (P[ix + iy * nx + iz * nx * ny] - K[ix + iy * nx + iz * nx * ny] * dt * diffV) ;
		Pf[ix + iy * nx + iz * nx * ny] = (P[ix + iy * nx + iz * nx * ny]  * mask[ix + iy * nx + iz * nx * ny] ) ;
		Ps[ix + iy * nx + iz * nx * ny] = (P[ix + iy * nx + iz * nx * ny] * (1-mask[ix + iy * nx + iz * nx * ny])) ;
		tau_xx[ix + iy * nx + iz * nx * ny] = ((DAT)1.0 / ((DAT)2.0 * G[ix + iy * nx + iz * nx * ny] * dt) * tau_xx[ix + iy * nx + iz * nx * ny] + diffVxx - (DAT)1.0 / (DAT)3.0 * diffV) * ((DAT)1.0 / ((DAT)1.0 / ((DAT)2.0 * G[ix + iy * nx + iz * nx * ny] * dt) + (DAT)1.0 / ((DAT)2.0 * eta[ix + iy * nx + iz * nx * ny]))) ;
		tau_yy[ix + iy * nx + iz * nx * ny] = ((DAT)1.0 / ((DAT)2.0 * G[ix + iy * nx + iz * nx * ny] * dt) * tau_yy[ix + iy * nx + iz * nx * ny] + diffVyy - (DAT)1.0 / (DAT)3.0 * diffV) * ((DAT)1.0 / ((DAT)1.0 / ((DAT)2.0 * G[ix + iy * nx + iz * nx * ny] * dt) + (DAT)1.0 / ((DAT)2.0 * eta[ix + iy * nx + iz * nx * ny]))) ;
		tau_zz[ix + iy * nx + iz * nx * ny] = ((DAT)1.0 / ((DAT)2.0 * G[ix + iy * nx + iz * nx * ny] * dt) * tau_zz[ix + iy * nx + iz * nx * ny] + diffVzz - (DAT)1.0 / (DAT)3.0 * diffV) * ((DAT)1.0 / ((DAT)1.0 / ((DAT)2.0 * G[ix + iy * nx + iz * nx * ny] * dt) + (DAT)1.0 / ((DAT)2.0 * eta[ix + iy * nx + iz * nx * ny]))) ;
	}
	if (ix > 0 && ix < nx && iy > 0 && iy < ny && iz < nz) {
		DAT G_av4 = (DAT)1.0 / ((DAT)1.0 / (DAT)4.0 * ((DAT)1.0 / G[ix - 1 + (iy - 1) * nx + iz * nx * ny] + (DAT)1.0 / G[ix + iy * nx + iz * nx * ny] + (DAT)1.0 / G[ix - 1 + iy * nx + iz * nx * ny] + (DAT)1.0 / G[ix + (iy - 1) * nx + iz * nx * ny]));
		DAT eta_av4 = (DAT)1.0 / ((DAT)1.0 / (DAT)4.0 * ((DAT)1.0 / eta[ix - 1 + (iy - 1) * nx + iz * nx * ny] + (DAT)1.0 / eta[ix + iy * nx + iz * nx * ny] + (DAT)1.0 / eta[ix - 1 + iy * nx + iz * nx * ny] + (DAT)1.0 / eta[ix + (iy - 1) * nx + iz * nx * ny]));
		DAT diffVxy = diffVx_y;
		DAT diffVyx = diffVy_x;
		tau_xy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = ((DAT)1.0 / (G_av4 * dt) * tau_xy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] + diffVxy + diffVyx) * ((DAT)1.0 / ((DAT)1.0 / (G_av4 * dt) + (DAT)1.0 / (eta_av4))) ;
	}
	if (ix > 0 && ix < nx && iy < ny && iz > 0 && iz < nz) {
		DAT G_av4 = (DAT)1.0 / ((DAT)1.0 / (DAT)4.0 * ((DAT)1.0 / G[ix - 1 + iy * nx + (iz - 1) * nx * ny] + (DAT)1.0 / G[ix + iy * nx + iz * nx * ny] + (DAT)1.0 / G[ix - 1 + iy * nx + iz * nx * ny] + (DAT)1.0 / G[ix + iy * nx + (iz - 1) * nx * ny]));
		DAT eta_av4 = (DAT)1.0 / ((DAT)1.0 / (DAT)4.0 * ((DAT)1.0 / eta[ix - 1 + iy * nx + (iz - 1) * nx * ny] + (DAT)1.0 / eta[ix + iy * nx + iz * nx * ny] + (DAT)1.0 / eta[ix - 1 + iy * nx + iz * nx * ny] + (DAT)1.0 / eta[ix + iy * nx + (iz - 1) * nx * ny]));
		DAT diffVxz = diffVx_z;
		DAT diffVzx = diffVz_x;
		tau_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = ((DAT)1.0 / (G_av4 * dt) * tau_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + diffVxz + diffVzx) * ((DAT)1.0 / ((DAT)1.0 / (G_av4 * dt) + (DAT)1.0 / (eta_av4))) ;
	}
	if (ix < nx && iy > 0 && iy < ny && iz > 0 && iz < nz) {
		DAT G_av4 = (DAT)1.0 / ((DAT)1.0 / (DAT)4.0 * ((DAT)1.0 / G[ix + (iy - 1) * nx + (iz - 1) * nx * ny] + (DAT)1.0 / G[ix + iy * nx + iz * nx * ny] + (DAT)1.0 / G[ix + (iy - 1) * nx + iz * nx * ny] + (DAT)1.0 / G[ix + iy * nx + (iz - 1) * nx * ny]));
		DAT eta_av4 = (DAT)1.0 / ((DAT)1.0 / (DAT)4.0 * ((DAT)1.0 / eta[ix + (iy - 1) * nx + (iz - 1) * nx * ny] + (DAT)1.0 / eta[ix + iy * nx + iz * nx * ny] + (DAT)1.0 / eta[ix + (iy - 1) * nx + iz * nx * ny] + (DAT)1.0 / eta[ix + iy * nx + (iz - 1) * nx * ny]));
		DAT diffVzy = diffVz_y;
		DAT diffVyz = diffVy_z;
		tau_yz[ix + iy * nx + iz * nx * (ny + 1)] = ((DAT)1.0 / (G_av4 * dt) * tau_yz[ix + iy * nx + iz * nx * (ny + 1)] + diffVyz + diffVzy) * ((DAT)1.0 / ((DAT)1.0 / (G_av4 * dt) + (DAT)1.0 / (eta_av4))) ;
	}




#undef diffVx_x
#undef diffVy_y
#undef diffVz_z
#undef diffVx_y
#undef diffVy_x
#undef diffVz_x
#undef diffVx_z
#undef diffVy_z
#undef diffVz_y
}

__global__ void compute_V(DAT* mask, DAT* Vxs, DAT* Vxf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* P, DAT* tau_xx, DAT* tau_yy, DAT* tau_zz, DAT* tau_xy, DAT* tau_xz, DAT* tau_yz, DAT* rho, const DAT dt, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, int istep) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
	int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
	int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
	CommOverlap();
#define diffP_x    ( (P[ ix    + iy     * nx     + iz     * nx     * ny   ]-P[(ix-1) + iy    * nx     + iz    * nx     * ny    ])/dx)
#define diffP_y    ( (P[ ix    + iy     * nx     + iz     * nx     * ny   ]-P[ ix    +(iy-1) * nx     + iz    * nx     * ny    ])/dy)
#define diffP_z    ( (P[ ix    + iy     * nx     + iz     * nx     * ny   ]-P[ ix    + iy    * nx     +(iz-1) * nx     * ny    ])/dz)
#define difftauxx_x  ( (tau_xx[ ix    + iy     * nx     + iz     * nx     * ny   ]-tau_xx[(ix-1) + iy    * nx     + iz    * nx     * ny    ])/dx)
#define difftauyy_y  ( (tau_yy[ ix    + iy     * nx     + iz     * nx     * ny   ]-tau_yy[ ix    +(iy-1) * nx     + iz    * nx     * ny    ])/dy)
#define difftauzz_z  ( (tau_zz[ ix    + iy     * nx     + iz     * nx     * ny   ]-tau_zz[ ix    + iy    * nx     +(iz-1) * nx     * ny    ])/dz)
#define difftauxy_y  ( (tau_xy[ ix    +(iy+1)  *(nx+1)  + iz     *(nx+1)  *(ny+1)]-tau_xy[ix + iy *(nx+1)  + iz *(nx+1)  *(ny+1) ])/dy)
#define difftauxy_x  ( (tau_xy[(ix+1) + iy     *(nx+1)  + iz     *(nx+1)  *(ny+1)]-tau_xy[ix + iy *(nx+1)  + iz *(nx+1)  *(ny+1) ])/dx)
#define difftauxz_x  ( (tau_xz[(ix+1) + iy     *(nx+1)  + iz     *(nx+1)  *(ny  )]-tau_xz[ix + iy *(nx+1)  + iz *(nx+1)  *(ny  ) ])/dx)
#define difftauxz_z  ( (tau_xz[ ix    + iy     *(nx+1)  +(iz+1)  *(nx+1)  *(ny  )]-tau_xz[ix + iy *(nx+1)  + iz *(nx+1)  *(ny  ) ])/dz)
#define difftauyz_y  ( (tau_yz[ ix    +(iy+1)  *(nx  )  + iz     *(nx  )  *(ny+1)]-tau_yz[ix + iy *(nx  )  + iz *(nx  )  *(ny+1) ])/dy)
#define difftauyz_z  ( (tau_yz[ ix    + iy     *(nx  )  +(iz+1)  *(nx  )  *(ny+1)]-tau_yz[ix + iy *(nx  )  + iz *(nx  )  *(ny+1) ])/dz)	
	if (iz < nz && iy < ny && ix>0 && ix < nx) {
		DAT diffpxx = diffP_x;
		DAT difftauxx = difftauxx_x;
		DAT difftauyx = difftauxy_y;
		DAT difftauzx = difftauxz_z;
		DAT rho_x = (DAT)0.5 * (rho[ix - 1 + iy * nx + iz * nx * ny] + rho[ix + iy * nx + iz * nx * ny]);
		DAT mask_x = (DAT)0.5 * (mask[ix - 1 + iy * nx + iz * nx * ny] + mask[ix + iy * nx + iz * nx * ny]);

		Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + dt / rho_x * (difftauxx - diffpxx + difftauyx + difftauzx)) ;
		Vxs[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny] * (1-mask_x);
		Vxf[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny] *mask_x;
	}
	if (iz < nz && iy > 0 && iy < ny && ix < nx) {
		DAT diffpyy = diffP_y;
		DAT difftauyy = difftauyy_y;
		DAT difftauxy = difftauxy_x;
		DAT difftauyz = difftauyz_z;
		DAT rho_y = (DAT)0.5 * (rho[ix + (iy - 1) * nx + iz * nx * ny] + rho[ix + iy * nx + iz * nx * ny]);
		Vy[ix + iy * nx + iz * nx * (ny + 1)] = (Vy[ix + iy * nx + iz * nx * (ny + 1)] + dt / rho_y * (difftauyy - diffpyy + difftauxy + difftauyz)) ;
	}
	if (iz > 0 && iz < nz && iy < ny && ix < nx) {
		DAT diffpzz = diffP_z;
		DAT difftauzz = difftauzz_z;
		DAT difftauzx = difftauxz_x;
		DAT difftauzy = difftauyz_y;
		DAT rho_z = (DAT)0.5 * (rho[ix + iy * nx + (iz - 1) * nx * ny] + rho[ix + iy * nx + iz * nx * ny]);
		Vz[ix + iy * nx + iz * nx * ny] = (Vz[ix + iy * nx + iz * nx * ny] + dt / rho_z * (difftauzz - diffpzz + difftauzx + difftauzy)) ;
	}
#undef diffP_x
#undef diffP_y
#undef diffP_z
#undef difftauxy_x
#undef difftauxy_y
#undef difftauxz_z
#undef difftauxz_x
#undef difftauyz_y
#undef difftauyz_z
}


int main(int argc, char* argv[]) {
	int i, it;
	set_up_grid();
	set_up_parallelisation();
	load(pa1, NPARS1, 1, "pa1.dat")
		double dx = pa1_h[0], dy = pa1_h[1], dz = pa1_h[2], dt = pa1_h[3];
	int irx1 = pa1_h[4], irx2 = pa1_h[5];
	int nt = pa1_h[6];
	int me0 = pa1_h[7], me1 =  pa1_h[8];


	if (me == 0) { printf("\n  ----------------------------------------------------------------------  "); }
	if (me == 0) { printf("\n  |   GPU-MPI 3D Poroelastic Wave Propagation in Heterogeneous Media   |  "); }
	if (me == 0) { printf("\n  ----------------------------------------------------------------------  \n\n"); }
	if (me == 0) { printf("Local sizes: Nx=%d, Ny=%d, Nz=%d, %d iterations. \n", nx, ny, nz, nt); }



	zeros(K, nx, ny, nz);
	def_sizes(K, nx, ny, nz);
	Load3(K, "K", "data", 0);

	zeros(G, nx, ny, nz);
	def_sizes(G, nx, ny, nz);
	Load3(G, "G", "data", 0);

	zeros(rho, nx, ny, nz);
	def_sizes(rho, nx, ny, nz);
	Load3(rho, "rho", "data", 0);

	zeros(eta, nx, ny, nz);
	def_sizes(eta, nx, ny, nz);
	Load3(eta, "eta", "data", 0);
	zeros(mask, nx, ny, nz);
	def_sizes(mask, nx, ny, nz);
	Load3(mask, "mask", "data", 0);

	load33(Src, nt, 1, 1, "Src.dat");

	// Initial arrays
	zeros(P, nx, ny, nz);
	zeros(Ps, nx, ny, nz);
	zeros(Pf, nx, ny, nz);
	zeros(tau_xx, nx, ny, nz);
	zeros(tau_yy, nx, ny, nz);
	zeros(tau_zz, nx, ny, nz);
	zeros(tau_xy, nx + 1, ny + 1, nz);
	zeros(tau_yz, nx, ny + 1, nz + 1);
	zeros(tau_xz, nx + 1, ny, nz + 1);

	zeros(Vx, nx + 1, ny, nz);
	zeros(Vxs, nx + 1, ny, nz);
	zeros(Vxf, nx + 1, ny, nz);
	zeros(Vy, nx, ny + 1, nz);
	zeros(Vz, nx, ny, nz + 1);

	zeros(Vx_rec1, nt, 1, 1);
	zeros(Vx_rec2, nt, 1, 1);
	zeros(Vx_rec3, nt, 1, 1);
	zeros(Vx_rec4, nt, 1, 1);
	zeros(Vx_rec5, nt, 1, 1);
	zeros(Vx_rec6, nt, 1, 1);
	zeros(Vx_rec7, nt, 1, 1);
	zeros(Vx_rec8, nt, 1, 1);

	// MPI sides    
	init_sides(Vx, nx + 1, ny, nz);
	init_sides(Vxs, nx + 1, ny, nz);
	init_sides(Vxf, nx + 1, ny, nz);
	init_sides(Vy, nx, ny + 1, nz);
	init_sides(Vz, nx, ny, nz + 1);


	// Preprocessing
	Nix = ((nx - 2) * dims[0]) + 2;
	Niy = ((ny - 2) * dims[1]) + 2;
	Niz = ((nz - 2) * dims[2]) + 2;




	if (me == 0) { printf("\ndt = %50.48f", dt); }
	if (me == 0) { printf("\ndx = %50.48f", dx); }



	double N3 = nx * ny * nz; double mem3 = (double)1e-9 * (double)N3 * sizeof(DAT);
	if (me == 0) { printf("Local sizes: Nx=%d, Ny=%d, Nz=%d, (%1.4f GB) %d iterations ...\n", nx, ny, nz, mem3 * 19.0, nt); }

	// Initial conditions
	int istep;

	// Action
	MPI_Barrier(topo_comm);
	for (it = 0;it < nt;it++) {
		if ((it % (int)5) == 0) { MPI_Barrier(topo_comm); } // TEST
		if (it == 50) { tic(); }
		// MPI overlap comm and compute
		compute_P << <grid, block >> > (mask_d, Vx_d, Vy_d, Vz_d, P_d, Ps_d, Pf_d,tau_xx_d, tau_yy_d, tau_zz_d, tau_xy_d, tau_xz_d, tau_yz_d, dt, K_d, G_d, eta_d, dx, dy, dz, nx, ny, nz, nt);
		cudaDeviceSynchronize();
		source << <grid, block >> > (coords_d, P_d, Src_d, it, nx, ny, nz, nt, mask_d); cudaDeviceSynchronize();
		for (istep = 0; istep < 2; istep++) {
			compute_V << <grid, block, 0, streams[istep] >> > (mask_d, Vxs_d, Vxf_d, Vx_d, Vy_d, Vz_d, P_d, tau_xx_d, tau_yy_d, tau_zz_d, tau_xy_d, tau_xz_d, tau_yz_d, rho_d, dt, dx, dy, dz, nx, ny, nz, istep);
			update_sides3(Vx, nx + 1, ny, nz, Vy, nx, ny + 1, nz, Vz, nx, ny, nz + 1)
		}
		cudaDeviceSynchronize();
		cudaMemset(Vx_rec1_d + it, 0, sizeof(DAT));
		cudaMemset(Vx_rec2_d + it, 0, sizeof(DAT));
		reciever << <grid, block >> > (coords_d, Vxs_d, Vxs_d, irx1, irx2, Vx_rec1_d, Vx_rec2_d, it, nx, ny, nz, nt); cudaDeviceSynchronize();
		cudaMemset(Vx_rec3_d + it, 0, sizeof(DAT));
		cudaMemset(Vx_rec4_d + it, 0, sizeof(DAT));
		reciever11<< <grid, block >> > (coords_d, Vxf_d, Vxf_d,irx1, irx2, Vx_rec3_d, Vx_rec4_d, it, nx, ny, nz, nt); cudaDeviceSynchronize();
		cudaMemset(Vx_rec5_d + it, 0, sizeof(DAT));
		cudaMemset(Vx_rec6_d + it, 0, sizeof(DAT));
		reciever22<< <grid, block >> > (coords_d, Vx_d, Vx_d,irx1, irx2, Vx_rec5_d, Vx_rec6_d, it, nx, ny, nz, nt); cudaDeviceSynchronize();
		cudaMemset(Vx_rec7_d + it, 0, sizeof(DAT));
		cudaMemset(Vx_rec8_d + it, 0, sizeof(DAT));
		reciever_src<< <grid, block >> > (coords_d, Ps_d,Pf_d,irx1, irx2, Vx_rec7_d, Vx_rec8_d, it, nx, ny, nz, nt); cudaDeviceSynchronize();


		if ((it % 2000) == 1 && (it > 1999)) {
			if (me == 0) { printf("\nit=%05d > ", it); fflush(stdout); }
		}
	}




	tim("Performance", Nix * Niy * Niz * (nt - 50) * 42 * PRECIS / (1e9)); // timer test
	// printf("Process %d used GPU with id %d.\n",me,gpu_id);

	if (me == 0) { printf("\nnx = %d", nx); }
	if (me == 0) { printf("\nNix = %d", Nix); }
	if (me == 0) { printf("\nNiy = %d", Niy); }
	if (me == 0) { printf("\nNiz = %d", Niz); }
	if (me == 0) { printf("\nny = %d", ny); }
	if (me == 0) { printf("\nnz = %d", nz); }
	if (me == 0) { printf("\ndx = %50.48f", dx); }
	if (me == 0) { printf("\ndy = %50.48f", dy); }
	if (me == 0) { printf("\ndz = %50.48f", dz); }
	if (me == 0) { printf("\ndt = %50.48f", dt); }
	if (me == 0) { printf("\nnx = %d\n", nx); }
	// save start
	save_info();
	save_coords();
	gather(Vx_rec1, nt, 1, 1);
	gather(Vx_rec2, nt, 1, 1);
	gather(Vx_rec3, nt, 1, 1);
	gather(Vx_rec4, nt, 1, 1);
	gather(Vx_rec5, nt, 1, 1);
	gather(Vx_rec6, nt, 1, 1);
	gather(Vx_rec7, nt, 1, 1);
	gather(Vx_rec8, nt, 1, 1);
	for (int it = 0; it < nt; it++) {
		Vx_rec1_h[it] /= (DAT)(ny * nz);
		Vx_rec2_h[it] /= (DAT)(ny * nz);
		Vx_rec3_h[it] /= (DAT)(ny * nz);
		Vx_rec4_h[it] /= (DAT)(ny * nz);
		Vx_rec5_h[it] /= (DAT)(ny * nz);
		Vx_rec6_h[it] /= (DAT)(ny * nz);
		Vx_rec7_h[it] /= (DAT)(ny * nz);
		Vx_rec8_h[it] /= (DAT)(ny * nz);
	}
	//SaveArrayold(Src ,"Src");
	if (me == me0) { save_array(Vx_rec1_h, nt, 1, 1, "Vxs_rec1"); };
	if (me == me1) { save_array(Vx_rec2_h, nt, 1, 1, "Vxs_rec2"); };
	if (me == me0) { save_array(Vx_rec3_h, nt, 1, 1, "Vxf_rec3"); };
	if (me == me1) { save_array(Vx_rec4_h, nt, 1, 1, "Vxf_rec4"); };
	if (me == me0) { save_array(Vx_rec5_h, nt, 1, 1, "Vx_rec5"); };
	if (me == me1) { save_array(Vx_rec6_h, nt, 1, 1, "Vx_rec6"); };
	if (me == me0) { save_array(Vx_rec7_h, nt, 1, 1, "srcs"); };
	if (me == me1) { save_array(Vx_rec8_h, nt, 1, 1, "srcf"); };
// 	SaveArray(P, nx, ny, nz, "P");
// 	SaveArray(Vx, nx, ny, nz, "Vx");
	free_all(P);
	free_all(Vx);
	free_all(Vy);
	free_all(Vz);
	free_all(tau_xx);
	free_all(tau_yy);
	free_all(tau_xy);
	free_all(tau_yz);
	free_all(tau_xz);
	free_all(tau_zz);
	free_all(Vx_rec1);
	free_all(Vx_rec2);
	// MPI
	free_sides(Vx);
	free_sides(Vy);
	free_sides(Vz);



	clean_cuda();
	MPI_Finalize();
	return 0;
}
