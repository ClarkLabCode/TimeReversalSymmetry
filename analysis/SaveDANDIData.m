clear
all_symmetries = {'T', 'C', 'TC', 'TX', 'CX', 'XTC', 'C,T,CT', ...
    'C,XT,XCT', 'T,XC,XCT', 'XC,XT,CT', 'nosymm_1', 'nosymm_2'};
stim_number = 1;

folder = fileparts(which(mfilename));
filepath = [folder, '/../analysis/DANDI_Data/'];
addpath([folder, '/..']);
addpath(genpath([folder, '/../lib/matlab']));


%need to eventually combine the stimuli statistics (net motion, etc) with
%responses. We don't actually care about the individual time traces (?) -
%we just care about the mean turning, for each fly. (indflymeanturning)


% Name of your stimulus
for num = 1:length(all_symmetries)
    
    symmetries = all_symmetries{num}
    
    
    stim = strcat('movingEdges_', symmetries);
    % Genotype of your flies
    genotype = 'Nathan_IsoD1';
    %date
    %year = 'Combined';
    %date = '07_27';
    
    % Get the path to the data directory on your computer
    sysConfig = GetSystemConfiguration;
    % concatenate those to create the path
    dataPath = [sysConfig.dataPath,'/',genotype,'/',stim, ...
        ..., '/' date ...
        ];
   
    %'figLeg',figLeg
    
    % Run the thing
    stimulus_name_1 = ['S_' num2str(stim_number)];
    stimulus_name_2 = ['S_' num2str(stim_number + 1)];
    epochNames = {'gray interleave', 'rightwards sine wave grating', 'leftwards sine wave grating', stimulus_name_1, ['\chi ' stimulus_name_1], ['\Theta ' stimulus_name_1], ['\chi \Theta ' stimulus_name_1], ['\Gamma ' stimulus_name_1], ['\chi \Gamma ' stimulus_name_1], ['\Gamma \Theta ' stimulus_name_1], ['\chi \Gamma \Theta ' stimulus_name_1], stimulus_name_2, ['\chi ' stimulus_name_2], ['\Theta ' stimulus_name_2], ['\chi \Theta ' stimulus_name_2], ['\Gamma ' stimulus_name_2], ['\chi \Gamma ' stimulus_name_2], ['\Gamma \Theta ' stimulus_name_2], ['\chi \Gamma \Theta ' stimulus_name_2]};
    epochNames
    [flyResp,epochs,params,stim,flyIds,rigIds,numTotalFlies, rawResp,rawStim,paramPaths,finalPaths,runDetails] = BuildDANDIDataset(symmetries, dataPath, filepath, 'epochNames', epochNames);
    
    
    %,...'figLeg',figLeg

    stim_number = stim_number + 2;
    
end