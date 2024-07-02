function [flyResp,epochs,params,stim,flyIds,rigIds,numTotalFlies] = ReadBehavioralData(dataFolders,varargin)

    %% Reads behavioral data, preprocess them and provide it to RunAnalysis
    % to be fed into individual analysis functions
    % commented by RT (1/16/21)

    % Default parameters
    blacklist = [];
    removeFlies = [];
    removeNonBehaving = 1;
    dataFromRig = [];
    rigSize = 5;
    runNumber = [];
    
    % Load all optional input arguments
    % varargin passed by RunAnalysis w/o modification
    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end

    % get data from file in format [time, [dx dy], files]
    [rawResp,rawStim,params,finalPaths,runDetails] = LoadData(dataFolders,blacklist,dataFromRig,runNumber);

    % dayRig behaviour records extra data after stimulus ends, leading to
    % extra epochs. I cut them off based on the response data -BA
    %if strcmp(runDetails{1,1}.rigName, 'dayRig')
    %    rawStim = rawStim(1:length(rawResp),:,:);
    %end 
    % reorder data into [time, flies, [dy dy]]
    [respDPI, stimAll, epochsAll, mouseReads] = GetTimeSeries(rawResp,rawStim);
    % rename this, convert from dpi to deg/sec and mm/sec
    flyRespAll = ConvertMouseFromDPI(respDPI,mouseReads);
    % flyRespAll has dimensions (response at each frame, 5 stations * # of experiments, dx for turning/dy for walking) -BA I think???

    % get flyIds
    singleFlyIds = GetFlyIds(runDetails);

    flyIds = RepFlyIds(singleFlyIds,rigSize);

    % remove nonbehaving/dead flies
    % get list of flies that behave
    if(removeNonBehaving)
        [selectedFlies,numTotalFlies] = GetResponsiveFlies(flyRespAll,epochsAll,flyIds);
    else
        selectedFlies = true(1,size(flyRespAll,2));
        numTotalFlies = size(flyRespAll,2);
    end 
    
    rigIds = repmat(1:5,[1 size(rawResp,3)]);
    
    
    %if strcmp(runDetails{1,1}.rigName, 'dayRig')
        % force analysis of the only fly on the dayRig
     %   selectedFlies = [true, false, false, false, false, true, false, false, false, false]; % fix this! should repeat [TFFFF] for every experiment
     %   numTotalFlies = size(flyRespAll,2)/5; % -BA
        
    %else
        selectFlies = ones(size(selectedFlies));
        selectFlies(removeFlies) = 0;
        
        numTotalFlies = numTotalFlies-(sum(selectedFlies)-sum(selectedFlies&selectFlies));
        
        selectedFlies = selectedFlies & selectFlies;
    %end 
   
    
    if(~any(selectedFlies))
        error('No flies selected / no flies behaved');
    end
    
    % update resp matrix with behaving flies
    flyRespSelected = flyRespAll(:,selectedFlies,:);
    
    flyIds = flyIds(selectedFlies);
    rigIds = rigIds(selectedFlies);
    
    % update time/fly epoch matrix to only include behaving flies
    epochsSelected = epochsAll(:,selectedFlies);

    % update stim matrix with behaving flies
    stimSelected = stimAll(:,:,selectedFlies);

    numFlies = size(flyRespSelected,2);

    flyResp = cell(1,numFlies);
    epochs = cell(1,numFlies);
    stim = cell(1,numFlies);

    for ff = 1:numFlies
        flyResp{ff} = flyRespSelected(:,ff,:);
        epochs{ff} = epochsSelected(:,ff);
        stim{ff} = stimSelected(:,:,ff);
    end
end