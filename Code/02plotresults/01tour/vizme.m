% Plot results used in the paper:
% comparison of analytical models, dynamic simulations,
% and quasi-static solutions for P-wave modulus and attenuation.
%
% Required files:
%   result_pizza11_vn.mat
%   Pwave_pizza.txt
%   Q_pizza.txt

clear;
clc;
close all;
format compact;

%% Dry stiffness tensors

Cij = [
    89.4676   6.2926   6.9382        0        0        0
     6.2926  89.4676   6.9382        0        0        0
     6.9382   6.9382  86.2973        0        0        0
          0        0        0  40.5278        0        0
          0        0        0        0  40.5279        0
          0        0        0        0        0  41.3184
] * 1e9;

Cij_cr = [
    88.9343   5.8032   4.1981        0        0        0
     5.8032  88.9343   4.1981        0        0        0
     4.1981   4.1981  53.7798        0        0        0
          0        0        0  32.8424        0        0
          0        0        0        0  32.8424        0
          0        0        0        0        0  41.2073
] * 1e9;

Sij = inv(Cij);
Sij_cr = inv(Cij_cr);

Z_N = Sij_cr(3,3) - Sij(3,3);
Z_T = Sij_cr(4,4) - Sij(4,4);

Z_N_reference = Z_N;
Z_T_reference = Z_T;

%% Geometry and material properties

L = 0.32;
crackThickness = 0.005;
crackRadius = 0.10;
torusMinorRadius = 0.020;
torusMajorRadius = 0.12;

modelVolume = L^3;
stiffPoreVolume = ...
    pi * torusMinorRadius^2 * (2 * pi * torusMajorRadius);
crackVolume = pi * crackRadius^2 * crackThickness;

totalPorosity = ...
    (stiffPoreVolume + crackVolume) / modelVolume;
stiffPorosity = stiffPoreVolume / modelVolume;
crackPorosity = crackVolume / modelVolume;

aspectRatio = crackThickness / (2 * crackRadius);

K_g = 36.0e9;
K_f = 4.3e9;
eta = 1.414;

rhoSolid = 2700;
rhoFluid = 1040;
rhoBulk = ...
    totalPorosity * rhoFluid + ...
    (1 - totalPorosity) * rhoSolid;

%% Saturated crack compliance

Z_Nsat = Z_N / ...
    (1 + Z_N / ...
    (crackPorosity * (1 / K_f - 1 / K_g)));

Sij_sat = Sij;
Sij_sat(3,3) = Sij_sat(3,3) + Z_Nsat;
Sij_sat(4,4) = Sij_sat(4,4) + Z_T;
Sij_sat(5,5) = Sij_sat(5,5) + Z_T;

leakCorrection = Sij_sat(3,3) - Sij(3,3);
Z_N = Z_N - leakCorrection;

%% Frequency range

frequency = 10.^(2:0.1:8);
omega = 2 * pi * frequency;
nFrequency = numel(frequency);

%% Present analytical model

capacitance = stiffPoreVolume / K_f;

kaLow = ...
    sqrt(-3i * eta / K_f) / aspectRatio;

KfLowFrequency = ...
    pi * crackRadius^2 * crackThickness * K_f / ...
    (pi * crackRadius^2 * crackThickness + ...
    capacitance * K_f);

C_00 = KfLowFrequency;
C_11 = K_f;

branchExponent = 0.45;

criticalAngularFrequency = ...
    4 * sqrt(3) * aspectRatio^2 * ...
    sqrt(C_00) * sqrt(C_11) / (3 * eta);

cg = C_00;
cg1 = C_11;
cg3 = aspectRatio;
cg5 = eta;
cg7 = branchExponent;
cg9 = criticalAngularFrequency;

Ksi = ...
    -3/8 * (-cg1 + cg) * cg5 * ...
    exp( ...
    (-log(sin(cg7 * pi / 2)) ...
    + 2 * log(cg1 - cg) ...
    + (cg7 + 1) * log(cg9) ...
    - log(cg7) ...
    - log(cg) ...
    - 3 * log(cg1) ...
    + 2 * log(cg5) ...
    - 4 * log(cg3) ...
    - 6 * log(2) ...
    + 2 * log(3)) / ...
    (cg7 - 1)) / ...
    (cg1^2 * cg3^2 * cg7);

cg = C_00;
cg1 = C_11;
cg3 = K_f;
cg5 = aspectRatio;
cg9 = eta;
cg11 = branchExponent;
cg13 = Ksi;

tauBranch = ...
    -(3/8) * cg9 * cg13 * ...
    (cg^2 - 2 * cg * cg3 + cg3^2) / ...
    (cg3^2 * cg5^2 * cg11 * (-cg1 + cg));

KfPresent = ...
    C_11 - ...
    C_11 ./ ...
    (1 - Ksi + ...
    Ksi .* ...
    (1 + 1i * omega * tauBranch / Ksi^2).^branchExponent);

Z_N_present = ...
    Z_N ./ ...
    (1 + Z_N ./ ...
    (crackPorosity .* ...
    (1 ./ KfPresent - 1 / K_g)));

CijPresentDry = complex(zeros(6,6,nFrequency));

for iFrequency = 1:nFrequency
    complianceIncrement = zeros(6,6);
    complianceIncrement(3,3) = ...
        Z_N_present(iFrequency) + leakCorrection;
    complianceIncrement(4,4) = Z_T;
    complianceIncrement(5,5) = Z_T;

    CijPresentDry(:,:,iFrequency) = ...
        inv(Sij + complianceIncrement);
end

CijPresent = anisotropic_gassmann( ...
    CijPresentDry, K_g, K_f, stiffPorosity);

%% Collet and Gurevich model A

ka = ...
    sqrt(-3i * omega * eta / K_f) / aspectRatio;

KfModelA = ...
    (1 - 2 * besselj(1,ka) ./ ...
    (ka .* besselj(0,ka))) * K_f;

Z_N_modelA = ...
    Z_N ./ ...
    (1 + Z_N ./ ...
    (crackPorosity .* ...
    (1 ./ KfModelA - 1 / K_g)));

CijModelADry = complex(zeros(6,6,nFrequency));

for iFrequency = 1:nFrequency
    complianceIncrement = zeros(6,6);
    complianceIncrement(3,3) = Z_N_modelA(iFrequency);
    complianceIncrement(4,4) = Z_T;
    complianceIncrement(5,5) = Z_T;

    CijModelADry(:,:,iFrequency) = ...
        inv(Sij + complianceIncrement);
end

CijModelA = anisotropic_gassmann( ...
    CijModelADry, K_g, K_f, stiffPorosity);

%% Collet and Gurevich model B

Z_N = Z_N_reference;
Z_T = Z_T_reference;

KfModelB = ...
    -(1/8) * ka.^2 * K_f;

Z_N_modelB = ...
    Z_N ./ ...
    (1 + Z_N ./ ...
    (crackPorosity .* ...
    (1 ./ KfModelB - 1 / K_g)));

CijModelBDry = complex(zeros(6,6,nFrequency));

for iFrequency = 1:nFrequency
    complianceIncrement = zeros(6,6);
    complianceIncrement(3,3) = Z_N_modelB(iFrequency);
    complianceIncrement(4,4) = Z_T;
    complianceIncrement(5,5) = Z_T;

    CijModelBDry(:,:,iFrequency) = ...
        inv(Sij + complianceIncrement);
end

CijModelB = anisotropic_gassmann( ...
    CijModelBDry, K_g, K_f, stiffPorosity);

%% Extract analytical modulus and attenuation

C33Present = squeeze(CijPresent(3,3,:)) / 1e9;
C33ModelA = squeeze(CijModelA(3,3,:)) / 1e9;
C33ModelB = squeeze(CijModelB(3,3,:)) / 1e9;

modulusPresent = real(C33Present);
modulusModelA = real(C33ModelA);
modulusModelB = real(C33ModelB);

attenuationPresent = ...
    imag(C33Present) ./ real(C33Present);

attenuationModelA = ...
    imag(C33ModelA) ./ real(C33ModelA);

attenuationModelB = ...
    imag(C33ModelB) ./ real(C33ModelB);

%% Load dynamic and quasi-static results

load('result_pizza11_vn.mat', 'scale', 'Vx_CM', 'Q1');

[quasiStaticFrequencyModulus, quasiStaticModulus] = ...
    read_comsol_global_txt('Pwave_pizza.txt');

[quasiStaticFrequencyQ, quasiStaticAttenuation] = ...
    read_comsol_global_txt('Q_pizza.txt');

referenceFrequency = 1e1;

dynamicFrequency = ...
    scale(1:numel(Vx_CM)) * referenceFrequency;

dynamicVelocity = ...
    Vx_CM * sqrt(1e10 / 2700);

dynamicModulus = ...
    rhoBulk * dynamicVelocity.^2 / 1e9;

dynamicAttenuation = Q1;

%% Plot paper figure

figure('Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [3 2 18 18]);

layout = tiledlayout(2,1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

ax1 = nexttile(layout);
hold(ax1, 'on');

plot(ax1, frequency, modulusPresent, 's-', ...
    'Color', [0.90 0.45 0.00], ...
    'LineWidth', 3.0, ...
    'MarkerSize', 7, ...
    'MarkerFaceColor', [0.90 0.45 0.00]);

plot(ax1, frequency, modulusModelA, '--', ...
    'Color', [0.00 0.20 0.90], ...
    'LineWidth', 3.0);

plot(ax1, dynamicFrequency, dynamicModulus, '-o', ...
    'Color', [0.85 0.10 0.10], ...
    'LineWidth', 2.0, ...
    'MarkerSize', 16, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', [0.85 0.10 0.10]);

plot(ax1, quasiStaticFrequencyModulus, quasiStaticModulus, '-d', ...
    'Color', [0.55 0.20 0.75], ...
    'LineWidth', 2.5, ...
    'MarkerSize', 16, ...
    'MarkerFaceColor', [0.55 0.20 0.75], ...
    'MarkerEdgeColor', [0.55 0.20 0.75]);

set(ax1, ...
    'XScale', 'log', ...
    'FontSize', 14, ...
    'LineWidth', 1.5, ...
    'TickDir', 'in', ...
    'Layer', 'top', ...
    'XMinorGrid', 'on', ...
    'YMinorGrid', 'on');

grid(ax1, 'on');
box(ax1, 'on');

xlim(ax1, [1e3 1e7]);
ylim(ax1, [69 90]);

ylabel(ax1, 'P-wave modulus (GPa)', ...
    'FontSize', 15);

legend(ax1, ...
    {'Analytical solution, Alkhimenkov and Quintal', ...
     'Analytical solution, Collet and Gurevich', ...
     'Dynamic solution', ...
     'Quasi-static solution'}, ...
    'Location', 'northwest', ...
    'FontSize', 11, ...
    'Box', 'on');

ax2 = nexttile(layout);
hold(ax2, 'on');

plot(ax2, frequency, attenuationPresent, 's-', ...
    'Color', [0.90 0.45 0.00], ...
    'LineWidth', 3.0, ...
    'MarkerSize', 7, ...
    'MarkerFaceColor', [0.90 0.45 0.00]);

plot(ax2, frequency, attenuationModelA, '--', ...
    'Color', [0.00 0.20 0.90], ...
    'LineWidth', 3.0);

plot(ax2, dynamicFrequency, dynamicAttenuation, '-o', ...
    'Color', [0.85 0.10 0.10], ...
    'LineWidth', 2.0, ...
    'MarkerSize', 16, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', [0.85 0.10 0.10]);

plot(ax2, quasiStaticFrequencyQ, quasiStaticAttenuation, '-d', ...
    'Color', [0.55 0.20 0.75], ...
    'LineWidth', 2.5, ...
    'MarkerSize', 16, ...
    'MarkerFaceColor', [0.55 0.20 0.75], ...
    'MarkerEdgeColor', [0.55 0.20 0.75]);

set(ax2, ...
    'XScale', 'log', ...
    'YScale', 'log', ...
    'FontSize', 14, ...
    'LineWidth', 1.5, ...
    'TickDir', 'in', ...
    'Layer', 'top', ...
    'XMinorGrid', 'on', ...
    'YMinorGrid', 'on');

grid(ax2, 'on');
box(ax2, 'on');

xlim(ax2, [1e3 1e7]);
ylim(ax2, [1e-4 1]);

xlabel(ax2, 'Frequency (Hz)', ...
    'FontSize', 15);

ylabel(ax2, '1/Q', ...
    'FontSize', 15);

linkaxes([ax1 ax2], 'x');

%% Local functions

function CijSaturated = anisotropic_gassmann( ...
    CijDry, grainBulkModulus, fluidBulkModulus, porosity)

    nFrequency = size(CijDry, 3);
    CijSaturated = complex(zeros(size(CijDry)));

    for iFrequency = 1:nFrequency
        currentCij = CijDry(:,:,iFrequency);

        Kstar = ...
            sum(sum(currentCij(1:3,1:3))) / 9;

        alphaColumn = zeros(6,1);
        alphaRow = zeros(1,6);

        for i = 1:3
            alphaColumn(i) = ...
                1 - sum(currentCij(i,1:3)) / ...
                (3 * grainBulkModulus);

            alphaRow(i) = ...
                1 - sum(currentCij(1:3,i)) / ...
                (3 * grainBulkModulus);
        end

        M = grainBulkModulus / ...
            ((1 - Kstar / grainBulkModulus) ...
            - porosity * ...
            (1 - grainBulkModulus / fluidBulkModulus));

        CijSaturated(:,:,iFrequency) = ...
            currentCij + alphaColumn * alphaRow * M;
    end
end

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
