clear
all_symmetries = {'T', 'C', 'TC', 'TX', 'CX', 'XTC', 'C,T,CT', 'C,XT,XCT', 'T,XC,XCT', 'XC,XT,CT', 'nosymm_1', 'nosymm_2'};
stim_number = 1;

folder = fileparts(which(mfilename));

filepath = [folder, '/../analysis/indfly_turning/'];

addpath([folder, '/..']);
addpath(genpath([folder, '/../lib/matlab']));


%need to eventually combine the stimuli statistics (net motion, etc) with
%responses. We don't actually care about the individual time traces (?) -
%we just care about the mean turning, for each fly. (indflymeanturning)

%% Running "RunAnalysis"

% RunAnalysis is the outermost wrapper function for data analysis both for
% behavior and imaging datout. It expects "analysisFile" and "dataPath"
% arguments at least. 

% Name of your stimulus
for num = 1:length(all_symmetries)
    clear newIndFlyMeanTurning
    
    symmetries = all_symmetries{num};
    
    
    stim = strcat('movingEdges_', symmetries);
    % Genotype of your flies
    genotype = 'Nathan_IsoD1';
    %date
    %year = 'Combined';
    %date = '07_27';
    
    % Get the path to the data directory on your computer
    sysConfig = GetSystemConfiguration;
    % concatenate those to create the path
    dataPath = [sysConfig.dataPath,'/',genotype,'/',stim,...
        ..., '/' date ...
        ];
    %dataPath or HollyDataPath
    % Your analysis file (you can pass multiple analysis function names as a
    % cell)
    % "PlotTimeTraces" clips out peri-stimulus turning and walking speed time
    % traces, averages over repetitions / flies, and plot group mean time traces
    % with standard error around it.
    % Other analysis functions can be found under analysis/analysisFiles
    analysisFiles={'PlotTimeTraces_Nathan'};
    
    %figLeg={'sine_r',	'sine_l',	'Stim1', 	'Stim1 x',	'Stim1 t',	'Stim1 tx',	'Stim1 c',	'Stim1 cx',	'Stim1 ct',	'Stim1 xct',	'Stim2', 	'Stim2 x',	'Stim2 t',	'Stim2 tx',	'Stim2 c',	'Stim2 cx',	'Stim2 ct',	'Stim2 xct'}
    figLeg={'sine'	'Stim1', 'Stim1 t',	'Stim1 c' 'Stim1 ct', 'Stim2', 'Stim2 t', 'Stim2 c', 'Stim2 ct'}
    
    
    % Prepare input arguments for RunAnalysis function
    args = {'analysisFile',analysisFiles,...
            'dataPath',dataPath,...
            'combOpp',1,'figLeg',figLeg}; % combine left/right symmetric parameters? (defaulted to 1)
    %'figLeg',figLeg
    
    % Run the thing
    iso_out = RunAnalysis(args{:}, 'ttSnipShift', -1020);
    close all
    %,...'figLeg',figLeg
    %%
    
    timeXiso = iso_out.analysis{1}.timeX/1000; % converting ms to s
    
    meanmat_iso = iso_out.analysis{1}.respMatPlot;%mean turning response and walking response over time - one dimension is the stimulus
    semmat_iso = iso_out.analysis{1}.respMatSemPlot; %standard error - on
    
    %%
    
    %Integrate over time using individual fly data
    nFly = length(iso_out.analysis{1}.indFly); %the number of flies
    indmat = [];
    for ff = 1:nFly
        % needs some reformatting from cell to matrix...
        thisFlyMat = cell2mat(permute(iso_out.analysis{1}.indFly{ff}.p8_averagedRois.snipMat,[3,1,2]));
        indmat(:,:,ff) = thisFlyMat(:,:,1); % only care about turning here...
    end
    
    size(indmat)
    indmat;
    %third dimension is each fly.

    startTime = 0.05;
    endTime = 3.05;
    
    indeces = find(timeXiso >= startTime & timeXiso < endTime);
    numValues = length(indeces);
    indFlyMeanTurning = permute(sum(indmat(indeces, :, :))./numValues, [3, 2, 1]); %integration step
    
    newIndFlyMeanTurning(1, :, 1:4) = indFlyMeanTurning(:, 2:5);
    newIndFlyMeanTurning(2, :, 1:4) = indFlyMeanTurning(:, 6:9);
    
    
    figLegs = {'Stim1', 'Stim1 t', 'Stim1 c', 'Stim1 ct',...
        'Stim2', 'Stim2 t', 'Stim2 c', 'Stim2 ct'};
    
    
    for i = 1:2
        fig = figure;
        categories = categorical(figLegs(4*i-3:4*i));
        categories = reordercats(categories, cellstr(categories)');
        %bar(categories, squeeze(newMeanTurning(i,:,:)));
    
        hold on
    
        s = scatter(categories, squeeze(newIndFlyMeanTurning(i,:,:)).',  'ok', 'filled');
        alpha(s, 0.3);
    
        hold off
        T{i} = array2table(squeeze(newIndFlyMeanTurning(i, :, :)), 'VariableNames', {'S', 'S_t', 'S_c', 'S_ct'});

        writetable(T{i}, [filepath 'S' num2str(stim_number) '_' symmetries '_' num2str(i) '.csv']);



        stim_number= stim_number + 1;
        
    end
end