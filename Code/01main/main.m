clc;
clear;
close all;
format compact;

%% ========================================================================
% MPI-GPU dynamic simulation for numerical dispersion and attenuation
%
% Required files and functions:
%   Results.mat
%   fracture_plus_4spheres.mat
%   init_data3D_v25.m
%   read_res.m
%   compute_Vp_Q_from_Vx.m
%   NS.m
%   runme_volta.sh
%
% Numerical outputs:
%   phaseVelocity : numerical phase velocity
%   attenuation   : numerical inverse quality factor, 1/Q
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('MPI-GPU numerical dispersion and attenuation simulation\n');
fprintf('============================================================\n');

%% 1. Load geometry and auxiliary model data

load('fracture_plus_4spheres.mat');

[nx, ny, nz] = size(volumeData);

fprintf('Grid size              : %d x %d x %d\n', nx, ny, nz);
fprintf('Fluid volume fraction  : %.6f\n', mean(volumeData(:) ~= 0));

%% 2. MPI decomposition

dims_x = 4;
dims_y = 1;
dims_z = 1;

if mod(nx, dims_x) ~= 0 || mod(ny, dims_y) ~= 0 || mod(nz, dims_z) ~= 0
    error('The global grid must be divisible by the MPI decomposition.');
end

nx_l = nx / dims_x;
ny_l = ny / dims_y;
nz_l = nz / dims_z;

OVERLENGTH_X = 2 * (dims_x - 1);
OVERLENGTH_Y = 2 * (dims_y - 1);
OVERLENGTH_Z = 2 * (dims_z - 1);

nx_write = nx - OVERLENGTH_X;
ny_write = ny - OVERLENGTH_Y;
nz_write = nz - OVERLENGTH_Z;

if nx_write <= 0 || ny_write <= 0 || nz_write <= 0
    error('Invalid overlap-adjusted model dimensions.');
end

machineformat = 'n';
PRECIS = 8;

Total_GPU = dims_x * dims_y * dims_z;
GPU_x = dims_x;
GPU_y = dims_y;
GPU_z = dims_z;

NBX = fix(nx_l / 32);
NBY = fix(ny_l / 4);
NBZ = fix(nz_l / 8);

if NBX < 1 || NBY < 1 || NBZ < 1
    error('The local grid is too small for the selected CUDA block size.');
end

fprintf('MPI decomposition      : %d x %d x %d\n', dims_x, dims_y, dims_z);
fprintf('Total GPUs             : %d\n', Total_GPU);
fprintf('Local grid size        : %d x %d x %d\n', nx_l, ny_l, nz_l);

%% 3. Numerical control parameters

modulusScale = 1e-10;

Lx = 1.0;
dx = Lx / (nx - 1);
dy = dx;
dz = dx;

sourceFrequency = 1e1;
frequencyIndex = log10(sourceFrequency);

scale = 10.^(-2:0.5:1) * 1e5;
nTests = numel(scale);

sourceCycles = 3;
sourceTolerance = 1e-3;

phaseVelocity = nan(nTests, 1);
attenuation = nan(nTests, 1);
timeStep = nan(nTests, 1);
numberOfSteps = nan(nTests, 1);

fprintf('Source frequency       : %.6e Hz\n', sourceFrequency);
fprintf('Number of tests        : %d\n', nTests);
fprintf('Grid spacing           : %.6e m\n', dx);

%% 4. Material properties

G1 = 44e9 * modulusScale;
K1 = 36e9 * modulusScale;
rho1 = 2700 / 2700;
eta1 = 1e20 * modulusScale;

G2 = 1e20 * modulusScale;
K2 = 4.3e9 * modulusScale;
rho2 = 1040 / 2700;
eta2 = 1.414 * modulusScale;

[Vps, ~] = NS(K1, G1, rho1, eta1, sourceFrequency);

if ~isfinite(Vps) || Vps <= 0
    error('Invalid solid-wave velocity returned by NS.');
end

fprintf('Reference wave velocity: %.6e\n', Vps);

%% 5. Build heterogeneous material arrays

G = G1 * ones(nx, ny, nz);
K = K1 * ones(nx, ny, nz);
rho = rho1 * ones(nx, ny, nz);
eta = eta1 * ones(nx, ny, nz);

fluidMask = (volumeData == 1) | (volumeData == 2);

G(fluidMask) = G2;
K(fluidMask) = K2;
rho(fluidMask) = rho2;

%% 6. Write static model fields

fprintf('\nWriting static material fields...\n');

init_data3D_v25( ...
    G(1:nx_write, 1:ny_write, 1:nz_write), ...
    'G', 0, nx_l, ny_l, nz_l, ...
    dims_x, dims_y, dims_z, machineformat, PRECIS);

init_data3D_v25( ...
    K(1:nx_write, 1:ny_write, 1:nz_write), ...
    'K', 0, nx_l, ny_l, nz_l, ...
    dims_x, dims_y, dims_z, machineformat, PRECIS);

init_data3D_v25( ...
    rho(1:nx_write, 1:ny_write, 1:nz_write), ...
    'rho', 0, nx_l, ny_l, nz_l, ...
    dims_x, dims_y, dims_z, machineformat, PRECIS);

init_data3D_v25( ...
    volumeData(1:nx_write, 1:ny_write, 1:nz_write), ...
    'mask', 0, nx_l, ny_l, nz_l, ...
    dims_x, dims_y, dims_z, machineformat, PRECIS);

fprintf('Static fields written successfully.\n');

%% 7. Check the external GPU launcher

pathToScript = fullfile(pwd, 'runme_volta.sh');

if ~isfile(pathToScript)
    error('GPU launcher was not found: %s', pathToScript);
end

%% 8. Run viscosity-scale simulations

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('Running numerical tests\n');
fprintf('------------------------------------------------------------\n');

for test_n = 1:nTests

    currentScale = scale(test_n);
    currentFluidViscosity = eta2 * currentScale;

    if currentScale > 10^(-0.5) * 1e5
        irx1 = fix(0.133 * Lx / dx);
        irx2 = fix(0.203 * Lx / dx);
        stabilityFactor = 0.15;
    else
        irx1 = fix(0.111 * Lx / dx);
        irx2 = fix(0.211 * Lx / dx);
        stabilityFactor = 0.50;
    end

    irx1 = max(1, min(nx - 1, irx1));
    irx2 = max(irx1 + 1, min(nx - 1, irx2));

    me1 = floor(irx1 / nx_l);
    me2 = floor(irx2 / nx_l);

    me1 = max(0, min(dims_x - 1, me1));
    me2 = max(0, min(dims_x - 1, me2));

    dtElastic = dx / Vps * stabilityFactor;

    dtViscous = ...
        (3 * rho2 * dx^2) / ...
        (4 * currentFluidViscosity + ...
        sqrt(9 * K2 * rho2 * dx^2 + ...
        16 * currentFluidViscosity^2)) * stabilityFactor;

    dt = min(dtElastic, dtViscous);

    if ~isfinite(dt) || dt <= 0
        error('Invalid time step for test %d.', test_n);
    end

    nt = max(1, fix(Lx / Vps / dt));

    timeStep(test_n) = dt;
    numberOfSteps(test_n) = nt;

    fb = (sourceCycles / (2 * pi * sourceFrequency))^2;
    t0 = ceil(sqrt(-fb * log(sourceTolerance)) / dt);

    timeVector = (0:nt-1).' * dt;
    sourceTime = timeVector - t0 * dt;
    src = cos(2 * pi * sourceFrequency * sourceTime) .* ...
          exp(-(sourceTime.^2) / fb);

    pa1 = [dx, dy, dz, dt, irx1, irx2, nt, me1, me2];

    write_binary_vector('pa1.dat', pa1);
    write_binary_vector('Src.dat', src);

    eta(:) = eta1;
    eta(fluidMask) = currentFluidViscosity;

    init_data3D_v25( ...
        eta(1:nx_write, 1:ny_write, 1:nz_write), ...
        'eta', 0, nx_l, ny_l, nz_l, ...
        dims_x, dims_y, dims_z, machineformat, PRECIS);

    command = sprintf( ...
        '"%s" %d %d %d %d %d %d %d %d', ...
        pathToScript, ...
        Total_GPU, GPU_x, GPU_y, GPU_z, ...
        numel(pa1), NBX, NBY, NBZ);

    fprintf('\n');
    fprintf('Test %02d/%02d\n', test_n, nTests);
    fprintf('Scale                  : %.6e\n', currentScale);
    fprintf('Fluid viscosity        : %.6e\n', currentFluidViscosity);
    fprintf('Receiver indices       : %d, %d\n', irx1, irx2);
    fprintf('Receiver MPI ranks     : %d, %d\n', me1, me2);
    fprintf('Time step              : %.6e s\n', dt);
    fprintf('Number of time steps   : %d\n', nt);

    exitStatus = system(command);

    if exitStatus ~= 0
        error( ...
            'MPI-GPU solver failed for test %d with exit status %d.', ...
            test_n, exitStatus);
    end

    receiverFile1 = sprintf('%d_Vx_rec5.res', me1);
    receiverFile2 = sprintf('%d_Vx_rec6.res', me2);

    if ~isfile(receiverFile1)
        error('Receiver file was not generated: %s', receiverFile1);
    end

    if ~isfile(receiverFile2)
        error('Receiver file was not generated: %s', receiverFile2);
    end

    [Vp_fast, inverseQ] = compute_Vp_Q_from_Vx( ...
        receiverFile1, ...
        receiverFile2, ...
        nt, ...
        irx1, ...
        irx2, ...
        dt, ...
        dx, ...
        sourceFrequency, ...
        false);

    if ~isfinite(Vp_fast)
        warning('Non-finite phase velocity at test %d.', test_n);
    end

    if ~isfinite(inverseQ)
        warning('Non-finite attenuation at test %d.', test_n);
    end

    phaseVelocity(test_n) = Vp_fast;
    attenuation(test_n) = abs(inverseQ);

    fprintf('Phase velocity          : %.8e\n', phaseVelocity(test_n));
    fprintf('Attenuation, 1/Q        : %.8e\n', attenuation(test_n));

    save( ...
        'NumericalDispersionAttenuation.mat', ...
        'scale', ...
        'phaseVelocity', ...
        'attenuation', ...
        'timeStep', ...
        'numberOfSteps', ...
        'sourceFrequency', ...
        'dx', ...
        'dy', ...
        'dz');
end

%% 9. Display the final numerical results

resultTable = table( ...
    scale(:), ...
    phaseVelocity(:), ...
    attenuation(:), ...
    timeStep(:), ...
    numberOfSteps(:), ...
    'VariableNames', { ...
    'Scale', ...
    'PhaseVelocity', ...
    'InverseQ', ...
    'TimeStep', ...
    'NumberOfSteps'});

fprintf('\n');
fprintf('============================================================\n');
fprintf('Final numerical results\n');
fprintf('============================================================\n');
disp(resultTable);

%% 10. Plot numerical dispersion and attenuation

figure('Name', 'Numerical dispersion and attenuation', 'Color', 'w');

subplot(2, 1, 1);
semilogx(scale, phaseVelocity, '-o', ...
    'LineWidth', 2, ...
    'MarkerSize', 7);
grid on;
box on;
xlabel('Viscosity scale');
ylabel('Phase velocity');
title('Numerical dispersion');

subplot(2, 1, 2);
loglog(scale, attenuation, '-o', ...
    'LineWidth', 2, ...
    'MarkerSize', 7);
grid on;
box on;
xlabel('Viscosity scale');
ylabel('1/Q');
title('Numerical attenuation');

save( ...
    'NumericalDispersionAttenuation.mat', ...
    'scale', ...
    'phaseVelocity', ...
    'attenuation', ...
    'timeStep', ...
    'numberOfSteps', ...
    'sourceFrequency', ...
    'dx', ...
    'dy', ...
    'dz', ...
    'resultTable');

fprintf('Results saved to NumericalDispersionAttenuation.mat\n');
fprintf('Simulation completed successfully.\n');

%% ========================================================================
% Local function
%% ========================================================================

function write_binary_vector(fileName, values)

    fileID = fopen(fileName, 'wb');

    if fileID < 0
        error('Unable to open file for writing: %s', fileName);
    end

    cleanupObject = onCleanup(@() fclose(fileID));

    numberWritten = fwrite(fileID, values(:), 'double');

    if numberWritten ~= numel(values)
        error('Incomplete binary write to file: %s', fileName);
    end
end
