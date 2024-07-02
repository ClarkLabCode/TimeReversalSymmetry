function [resp,stim,params,paths,runDetails] = LoadData(folders,blacklist,dataFromRig,runNumber)
    %set up structure D which will hold all the variables pertaining to the
    %data and data analysis
    
    if nargin<3
        dataFromRig = [];
    end
    
    sysConfig = GetSystemConfiguration();
    
    dataPath = sysConfig.dataPath;
    flipToMouseDelay = sysConfig.flipToMouseDelay;
    
    if isempty(folders)
        blacklist = [];
        
        paths.folders = UiPickFiles('FilterSpec',dataPath,'Prompt','Choose folders containing the files to be analyzed');
        D.folders = paths.folders;
    else
        if iscell(folders)
            paths.folders = folders;
        else
            paths.folders = {folders};
        end
    end
    
    if nargin<2
        blacklist = [];
    end
    
    paths.resp = cell(0,1);
    paths.stim = cell(0,1);
    paths.meta = cell(0,1);
    paths.param = cell(0,1);
    paths.runDetails = cell(0,1);
    
    if ~iscell(paths.folders)
        error('no files to analyze');
    end
    
    %go through each folder selected and grab the path to every resp and
    %stimData file
    for ii = 1:size(paths.folders,2)        
        respInSubDir = DirRec(paths.folders{ii},'respdata*')';
        paths.resp = cat(1,paths.resp,respInSubDir);
        
        stimInSubDir = DirRec(paths.folders{ii},'stimdata*')';
        paths.stim = cat(1,paths.stim,stimInSubDir);
        
        metaInSubDir = DirRec(paths.folders{ii},'metadata*')';
        paths.meta = cat(1,paths.meta,metaInSubDir);
        
        metaInSubDir = DirRec(paths.folders{ii},'chosenparams*')';
        paths.param = cat(1,paths.param,metaInSubDir);
    end
    
    if strcmp(paths.folders{1}(end-3:end),'.mat')
        paths = load(paths.folders{1});
        
        for pp = 1:length(paths.resp)
            paths.resp{pp} = fullfile(dataPath,paths.resp{pp});
        end
        
        paths.meta = cell(0,1);
    end
    
    if ~isempty(blacklist)
        if isempty(regexp(blacklist(1:3),'[A-z]\:\\','once'))
            blacklist = fullfile(dataPath,blacklist);
        end
        
        blacklist = load(blacklist);
        for bb = 1:length(blacklist.resp)
            blacklist.resp{bb} = fullfile(dataPath,blacklist.resp{bb});
            
            for pp = length(paths.resp):-1:1
                if strcmp(paths.resp{pp},blacklist.resp{bb})
                    paths.resp(pp) = [];
                    paths.stim(pp) = [];
                    paths.param(pp) = [];
                    paths.meta(pp) = [];
                end
            end
        end
    end
    
    rigName = cell(size(paths.meta));
    
    % remove all files not from the rig specified here
    if ~isempty(dataFromRig) && ~isequal(dataFromRig,'any')
        returnInt = 10;
        
        for pp = length(paths.meta):-1:1
            % find rig name
            stringToFind = 'rig name = ';
            strOffset = length(stringToFind); % size of string we're looking for
            metaFile = fileread(paths.meta{pp});
            rigNamePosition = strfind(metaFile,stringToFind)+strOffset;
            returnPosition = strfind(metaFile(rigNamePosition:end),returnInt);
            
            rigName{pp} = metaFile(rigNamePosition:rigNamePosition+returnPosition(1)-2);
            
            if ~strcmp(rigName{pp},dataFromRig)
                rigName(pp) = [];
                paths.resp(pp) = [];
                paths.stim(pp) = [];
                paths.param(pp) = [];
                paths.meta(pp) = [];
            end
        end
    end
    
    
    
    %% load runDetails
    runDetails = cell(length(paths.resp),1);
    
    for rr = 1:length(runDetails)
        runDetailsPath = fullfile(fileparts(paths.resp{rr}),'runDetails.mat');
        
        if exist(runDetailsPath,'file') == 2
            runDetails{rr} = load(runDetailsPath);
        else
            flyId = AssignFlyId(0);
            runDetails{rr}.flyId = flyId;
            save(runDetailsPath,'flyId');
        end
    end
    
    % remove all files not from the wrong run number
    if ~isempty(runNumber)
        for pp = length(paths.resp):-1:1
            if runDetails{pp}.runNumber ~= runNumber
                rigName(pp) = [];
                paths.resp(pp) = [];
                paths.stim(pp) = [];
                paths.param(pp) = [];
                paths.meta(pp) = [];
                runDetails(pp) = [];
            end
        end
    end
    
    D.paths = paths;
    %initialize figure array even if its unused
    D.figures = cell(0,1);
    D.data.params = cell(1,5*length(D.paths.resp));
    
    
    
    %% check if any files are not .mat and are still in .txt format
    % convert to .mat
    for ii = 1:length(D.paths.resp)
        loadedParams = load(D.paths.param{ii});
        
        if isfield(loadedParams,'stimParams')
            D.data.params((5*(ii-1)+1):(5*ii)) = {loadedParams.stimParams};
        else
            D.data.params((5*(ii-1)+1):(5*ii)) = {loadedParams.params};
        end
        
        if strcmp(D.paths.resp{ii}(end-2:end),'csv')
            respSaveTo = fullfile([fileparts(D.paths.resp{ii}) '\respdata.mat']);
            
            if exist(D.paths.resp{ii},'file')
                respData = csvread(D.paths.resp{ii});
                respNameChange = fullfile([fileparts(D.paths.resp{ii}) '\textRespData.csv']);
                save(respSaveTo,'respData');
                movefile(D.paths.resp{ii},respNameChange)
            end
            
            D.paths.resp{ii} = respSaveTo;
        end
        
        if strcmp(D.paths.stim{ii}(end-2:end),'csv')
            stimSaveTo = fullfile([fileparts(D.paths.stim{ii}) '\stimdata.mat']);
            
            if exist(D.paths.stim{ii},'file')
                stimData = csvread(D.paths.stim{ii},1,0);
                stimNameChange = fullfile([fileparts(D.paths.resp{ii}) '\textStimData.csv']);
                save(stimSaveTo,'stimData');
                movefile(D.paths.stim{ii},stimNameChange)
            end
            
            D.paths.stim{ii} = stimSaveTo;
        end
    end
    
    %% get file size
    
    respSize = zeros(length(D.paths.resp),2);
    stimSize = zeros(length(D.paths.resp),2);
    
    for ff = 1:length(D.paths.resp)
        tempRespSize = whos('-file',D.paths.resp{ff});
        respSize(ff,:) = tempRespSize.size;
        
        tempStimSize = whos('-file',D.paths.stim{ff});
        stimSize(ff,:) = tempStimSize.size;
    end
    
    sizeRespTime = max(respSize(:,1));
    sizeRespMeasures = max(respSize(:,2));
    
    minRespTime = min(respSize(:,1));
    minRespMeasures = min(respSize(:,2));
    
    sizeStimTime = max(stimSize(:,1));
    sizeStimMeasures = max(stimSize(:,2));
    
    if sizeRespTime - minRespTime > 10
        error('more than 10 frames difference between parameter files');
    end
    
    if sizeRespMeasures - minRespMeasures > 2
        error('respdatas have different numbers of columns');
    end

    sizeRespNumFiles = size(D.paths.resp,1);
    sizeStimNumFiles = size(D.paths.stim,1);
    
    %% read in data
    D.data.resp = zeros(sizeRespTime,sizeRespMeasures,sizeRespNumFiles);
    D.data.stim = zeros(sizeStimTime,sizeStimMeasures,sizeStimNumFiles);

    D.data.resp(:,18,:) = 1;
    
    %read in the data from the filepaths
    for ii = 1:size(D.data.resp,3)
        respTemp = load(D.paths.resp{ii});
        respTemp = respTemp.respData;
        D.data.resp(1:size(respTemp,1),1:size(respTemp,2),ii) = respTemp;
        
        stimTemp = load(D.paths.stim{ii});
        stimTemp = stimTemp.stimData;
        D.data.stim(1:size(stimTemp,1),1:size(stimTemp,2),ii) = stimTemp;
    end

    %if the read is delayed shift all data up one time point and delete the
    %last one
    D.data.resp(1:end-flipToMouseDelay,3:end,:) = D.data.resp(flipToMouseDelay+1:end,3:end,:);

    D.data.resp(end-flipToMouseDelay+1:end,:,:) = [];
    D.data.stim((end-flipToMouseDelay+1):end,:,:) = [];
    
    resp = D.data.resp;
    stim = D.data.stim;
    params = D.data.params;
    paths = D.paths;
end

% TODO: support for non-5 numbers of flies per file
% Get rid of the D
% Move nargin case outside