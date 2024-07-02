%%

folder = fileparts(which(mfilename));

save_filepath = [folder, '/../figures/fig1/'];
addpath([folder, '/..']);
addpath(genpath([folder, '/../lib/matlab']));

%%
%Margarida jitter - GOOD FIGURE - SELF SYMMETRY

figLeg={'sine right', 'Stationary sawtooth',  'Jittered sawtooth',  '30Hz, 5step', '40Hz, 5step',  '50Hz, 5step',  '60Hz, 5step'}
pairs = {[2 2], [3 3], [4 4], [5 5], [6 6], [7 7]};
stimuli_indices = {2, 3};
mode = 't_flip';

stimulus_duration = 2;

stim = 'sawtoothSpatial_lam45_C1_sinewave_5sec_jitter_variousFreqsAndSteps3';
genotype = 'IsoD1_margarida';
year = '2020';

xtPlot_year = '2023';

XTFlipFigureSelfSymmetry(figLeg, stimuli_indices, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year)
disp('done')
%%
%Margarida moving sawtooth 1 - 17 flies

xFlipPairs = {[1 2], [3 16], [5 14], [7 12], [9 10], [11 8], [13 6], [15 4]};


figLeg={'sine', 'v=-1,5', 'v=-1', 'v=-0,5', 'v=0', 'v=0,5', 'v=1', 'Sawtooth grating'}
pairs = {[8 2]};
mode = 't_flip';

stimulus_duration = 5;

stim = 'sawtoothSpatial_stat_mov_lam45_C1_sinewave_5sec_moreVels_2_numDeg1';
genotype = 'w_+;+;+(empty single)';
year = '2019';

xtPlot_year = '2023';


XTFlipFigure(figLeg, pairs, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year, 'xFlipPairs', xFlipPairs)
disp('done')

%%
figLeg={'Sine wave grating', 'div+ right', 'div- right',	'conv+ right','conv- right',	'2pt+ right',	'2pt- right',	'elbow+ pointing left',	'elbow- pointing left'}
pairs = {1};
mode = 'xt_flip';
stimulus_duration = 1;

stim = 'all_gliders';
% Genotype of your flies
genotype = 'Nathan_IsoD1';
%date
year = '2022';

xtPlot_year = '2024';

XTFlipFigureSelfSymmetry(figLeg, pairs, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year)
disp('done')
%%
figLeg={'sine r', 'dr,dr/dr,dr',	'Light rightward edge', 'dr,dr/ll,ll',	'dr,ll/ll,dr'}
pairs = {[3 2]};
mode = 'xt_flip';
stimulus_duration = 3;

stim = 'movingEdges_spatialTemporalIntegration';
% Genotype of your flies
genotype = 'Nathan_IsoD1';
%date
year = '2022';

xtPlot_year = '2024';

XTFlipFigure(figLeg, pairs, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year)
disp('done')

%%
figLeg={'sine r', 'dr,dr/dr,dr',	'Light rightward edge', 'Alternating edges',	'Opposing edges'}
pairs = {5, 4};
mode = 't_flip';
stimulus_duration = 3;

stim = 'movingEdges_spatialTemporalIntegration';
% Genotype of your flies
genotype = 'Nathan_IsoD1';
%date
year = '2022';

xtPlot_year = '2024';

XTFlipFigureSelfSymmetry(figLeg, pairs, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year)
disp('done')


%%
figLeg={'Uneven square wave grating',	'30_3_r',	'30_5_r',	'60_1_r',	'60_3_r',	'60_5_r', '120_1_r', '120_3_r',	'120_5_r'}

pairs = {[1 3]};
mode = 'xt_flip';
stimulus_duration = 1;
stim = 'UnevenSquareGratingScreen_60px_medium_1s';
% Genotype of your fl
genotype = 'Nathan_IsoD1';
%date
year = '2024';

xtPlot_year = '2024';

XTFlipFigure(figLeg, pairs, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year)
disp('done')
