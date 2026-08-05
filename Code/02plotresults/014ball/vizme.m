% Plot results used in the paper:
% comparison of the analytical, dynamic, and quasi-static solutions
% for P-wave modulus and attenuation.
%
% Required files:
%   result_pizza11_vn.mat
%   Pwave_4ball.txt
%   Q_4ball.txt

clear;
clc;
close all;

%% Dry stiffness tensors

Cij = [
    88.6765   6.7492   6.6540        0        0        0
     6.7492  88.6765   6.6540        0        0        0
     6.6540   6.6540  88.2143        0        0        0
          0        0        0  41.2718        0        0
          0        0        0        0  41.2718        0
          0        0        0        0        0  40.9640
] * 1e9;

Cij_cr = [
    88.4328   6.3825   6.2395        0        0        0
     6.3825  88.4328   6.2395        0        0        0
     6.2395   6.2395  76.7415        0        0        0
          0        0        0  38.2509        0        0
          0        0        0        0  38.2509        0
          0        0        0        0        0  40.7417
] * 1e9;

Sij = inv(Cij);
Sij_cr = inv(Cij_cr);

Z_N = Sij_cr(3,3) - Sij(3,3);
Z_T = Sij_cr(4,4) - Sij(4,4);

%% Geometry and material properties

L = 16;
crackThickness = 0.25;
crackRadius = 3.75;
poreRadius = 2;

modelVolume = L^3;
stiffPoreVolume = 4 * (4/3) * pi * poreRadius^3;
crackVolume = pi * crackRadius^2 * crackThickness;

totalPorosity = (stiffPoreVolume + crackVolume) / modelVolume;
stiffPorosity = stiffPoreVolume / modelVolume;
crackPorosity = crackVolume / modelVolume;

aspectRatio = crackThickness / (2 * crackRadius);

K_g = 36.0e9;
K_f = 4.3e9;
eta = 1.414;

rhoSolid = 2700;
rhoFluid = 1040;
rhoBulk = totalPorosity * rhoFluid + ...
          (1 - totalPorosity) * rhoSolid;

%% Saturated crack compliance

Z_Nsat = Z_N / ...
    (1 + Z_N / (crackPorosity * (1 / K_f - 1 / K_g)));

Sij_sat = Sij;
Sij_sat(3,3) = Sij_sat(3,3) + Z_Nsat;
Sij_sat(4,4) = Sij_sat(4,4) + Z_T;
Sij_sat(5,5) = Sij_sat(5,5) + Z_T;

leakCorrection = Sij_sat(3,3) - Sij(3,3);
Z_N = Z_N - leakCorrection;

%% Analytical solution

frequency = 10.^(-5:0.1:8);
omega = 2 * pi * frequency;

G_fluid = 1i * omega * eta;
layerArgument = ...
    sqrt(3 * G_fluid ./ (K_f + 4/3 * G_fluid)) / aspectRatio;

K_fluid_effective = ...
    K_f + 4/3 * G_fluid ...
    - (K_f - 2/3 * G_fluid).^2 ./ ...
      (K_f + 4/3 * G_fluid) .* ...
      tanh(layerArgument) ./ layerArgument;

Z_N_dynamic = Z_N ./ ...
    (1 + Z_N ./ ...
    (crackPorosity .* (1 ./ K_fluid_effective - 1 / K_g)));

nFrequency = numel(frequency);

Cij_dry_dynamic = complex(zeros(6,6,nFrequency));

for iFrequency = 1:nFrequency
    complianceIncrement = zeros(6,6);
    complianceIncrement(3,3) = ...
        Z_N_dynamic(iFrequency) + leakCorrection;
    complianceIncrement(4,4) = Z_T;
    complianceIncrement(5,5) = Z_T;

    Cij_dry_dynamic(:,:,iFrequency) = ...
        inv(Sij + complianceIncrement);
end

%% Anisotropic Gassmann fluid substitution

Cij_final = Cij_dry_dynamic;
K_star = complex(zeros(1,nFrequency));
alpha_i = complex(zeros(6,1,nFrequency));
alpha_j = complex(zeros(1,6,nFrequency));
M = complex(zeros(1,nFrequency));

for iFrequency = 1:nFrequency
    currentCij = Cij_dry_dynamic(:,:,iFrequency);

    K_star(iFrequency) = ...
        sum(sum(currentCij(1:3,1:3))) / 9;

    for i = 1:3
        alpha_i(i,1,iFrequency) = ...
            1 - sum(currentCij(i,1:3)) / (3 * K_g);

        alpha_j(1,i,iFrequency) = ...
            1 - sum(currentCij(1:3,i)) / (3 * K_g);
    end

    M(iFrequency) = K_g / ...
        ((1 - K_star(iFrequency) / K_g) ...
        - stiffPorosity * (1 - K_g / K_f));

    Cij_final(:,:,iFrequency) = ...
        currentCij + ...
        alpha_i(:,:,iFrequency) * ...
        alpha_j(:,:,iFrequency) * ...
        M(iFrequency);
end

analyticalModulus = ...
    real(squeeze(Cij_final(3,3,:))) / 1e9;

analyticalAttenuation = ...
    imag(squeeze(Cij_final(3,3,:))) ./ ...
    real(squeeze(Cij_final(3,3,:)));

%% Load dynamic and quasi-static results

load('result_pizza11_vn.mat', 'scale', 'Vx_CM', 'Q1');

[quasiStaticFrequencyModulus, quasiStaticModulus] = ...
    read_comsol_global_txt('Pwave_4ball.txt');

[quasiStaticFrequencyQ, quasiStaticAttenuation] = ...
    read_comsol_global_txt('Q_4ball.txt');

referenceFrequency = 1e1;

dynamicFrequency = ...
    scale(1:numel(Vx_CM)) * referenceFrequency;

dynamicVelocity = ...
    Vx_CM * sqrt(1e10 / 2700);

dynamicModulus = ...
    rhoBulk * dynamicVelocity.^2 / 1e9;

dynamicAttenuation = Q1;

%% Plot paper figure

figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [3 2 18 18]);

layout = tiledlayout(2,1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

ax1 = nexttile(layout);

semilogx(frequency, analyticalModulus, '-', ...
    'Color', [0.50 0.10 0.10], ...
    'LineWidth', 3.0);
hold on;

semilogx(dynamicFrequency, dynamicModulus, '-o', ...
    'Color', [0.85 0.10 0.10], ...
    'MarkerEdgeColor', [0.85 0.10 0.10], ...
    'MarkerFaceColor', 'w', ...
    'MarkerSize', 10, ...
    'LineWidth', 1.8);

semilogx(quasiStaticFrequencyModulus, quasiStaticModulus, '-d', ...
    'Color', [0.15 0.35 0.85], ...
    'MarkerEdgeColor', [0.15 0.35 0.85], ...
    'MarkerFaceColor', [0.15 0.35 0.85], ...
    'MarkerSize', 10, ...
    'LineWidth', 2.0);

grid on;
box on;
xlim([1e3 1e7]);
ylim([78 90]);
ylabel('P-wave modulus (GPa)', 'FontSize', 13);

legend( ...
    {'Analytical solution', ...
     'Dynamic solution', ...
     'Quasi-static solution'}, ...
    'Location', 'southeast', ...
    'FontSize', 10, ...
    'Box', 'on');

set(ax1, ...
    'FontSize', 12, ...
    'LineWidth', 1.1, ...
    'XMinorGrid', 'on', ...
    'YMinorGrid', 'on', ...
    'TickDir', 'out');

ax2 = nexttile(layout);

loglog(frequency, analyticalAttenuation, '-', ...
    'Color', [0.50 0.10 0.10], ...
    'LineWidth', 3.0);
hold on;

loglog(dynamicFrequency, dynamicAttenuation, '-o', ...
    'Color', [0.85 0.10 0.10], ...
    'MarkerEdgeColor', [0.85 0.10 0.10], ...
    'MarkerFaceColor', 'w', ...
    'MarkerSize', 10, ...
    'LineWidth', 1.8);

loglog(quasiStaticFrequencyQ, quasiStaticAttenuation, '-d', ...
    'Color', [0.15 0.35 0.85], ...
    'MarkerEdgeColor', [0.15 0.35 0.85], ...
    'MarkerFaceColor', [0.15 0.35 0.85], ...
    'MarkerSize', 10, ...
    'LineWidth', 2.0);

grid on;
box on;
xlim([1e3 1e7]);
ylim([1e-5 1e0]);
xlabel('Frequency (Hz)', 'FontSize', 13);
ylabel('1/Q', 'FontSize', 13);

set(ax2, ...
    'FontSize', 12, ...
    'LineWidth', 1.1, ...
    'XMinorGrid', 'on', ...
    'YMinorGrid', 'on', ...
    'TickDir', 'out');

linkaxes([ax1 ax2], 'x');

%% Local function

function [frequency, value] = read_comsol_global_txt(fileName)

    fileID = fopen(fileName, 'r');

    if fileID < 0
        error('Cannot open file: %s', fileName);
    end

    cleanupObject = onCleanup(@() fclose(fileID));

    frequency = [];
    value = [];

    while ~feof(fileID)
        currentLine = strtrim(fgetl(fileID));

        if isempty(currentLine) || startsWith(currentLine, '%')
            continue;
        end

        currentData = sscanf(currentLine, '%f %f');

        if numel(currentData) == 2
            frequency(end+1,1) = currentData(1);
            value(end+1,1) = currentData(2);
        end
    end

    if isempty(frequency)
        error('No valid two-column data were found in %s.', fileName);
    end
end
