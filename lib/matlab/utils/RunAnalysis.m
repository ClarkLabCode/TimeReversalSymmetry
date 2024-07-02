function D = RunAnalysis(varargin)
    %% this was written by Matt Creamer the Great (praise him) and it is a
    % golden piece of glory in an otherwise dull world. This .m will call
    % the analysis function and determine if a meta analysis is performed
    % as well
    
    %% Comments added by RT for posterity (1/16/21)
    % Note: this function is written to deal with both behavioral and
    % imaging data but master and twoPhoton versions have deviated from
    % each other significantly since 2017. Consider merging at some point?
    
    % In 2p version of this function, dataPath etc. are declared as
    % persistent variables as follows:
    % persistent dataPath flyResp epochs params stim argumentOut dataRate dataType interleaveEpoch numTotalFlies flyIdsSel vararginPrev
    
    % List of relevant flags:
    % analzeEmpty
    % getUniqueFliesFlag
    % repeatDataPreprocess
    % varsToIgnoreForReprocess
    
    % Preaparing output structures etc.
    D = [];
    analysisFile = [];
    selectedEpochs = {'' ''}; % only used for imaging?
    dataPath = '';

    % Read sysConfig.csv file to find data locations etc. 
    sysConfig = GetSystemConfiguration();
    
    getUniqueFliesFlag = true;
    repeatDataPreprocess = true; % unused in this version
    
    % Read all optional input arguments
    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end

    argumentOut = cell(0,1);
    
    % randomize seedstate
    seedState = rng('shuffle');
    
    if isempty(D)
        
        %% get path to the data with GUI if it wasn't provided explicitly
        dataFolderPath = sysConfig.dataPath;
        
        if isempty(dataPath)
            dataPath = UiPickFiles('FilterSpec',dataFolderPath,'Prompt','Choose folders containing the files to be analyzed');
        else
            if ~iscell(dataPath)
                dataPath = {dataPath};
            end
        end
        
        %% Check data type (behavior / imaging / ephys)
        behavioralData = 1;
        imagingData = 0;
        ephysData = 0;
        fullDirectory = cell(1,length(dataPath));
        
        for dd = 1:length(dataPath)
            if isempty(regexp(dataPath{dd}(1:3),'[A-z]\:[\\\/]','once')) && isempty(regexp(dataPath{dd}(1:2),'\\\\|\/[A-z]','once')) &&  (dataPath{dd}(1) ~= '/')
                fullDirectory{dd} = fullfile(dataFolderPath,dataPath{dd});
            else
                fullDirectory{dd} = dataPath{dd};
            end
        end
        
        % If this is an imaging data, the direcotry should have
        % alignedMovie.mat
        if ~isempty(DirRec(fullfile(fullDirectory{1},'alignedMovie.mat')))
            behavioralData = 0;
            imagingData = 1;
        end
        
        % If this is an ephys data, the directory should have .abf file (we
        % don't really do this anymore)
        if ~isempty(DirRec(fullDirectory{1},'.abf'))
            behavioralData = 0;
            ephysData = 1;
        end
        
        % Otherwise this function assumes it is a behavioral data
        
        %% organize the input data
        if behavioralData
            disp('Analyzing behavioral data...');
            [flyResp,epochs,params,stim,flyIds,rigIds,numTotalFlies] = ReadBehavioralData(fullDirectory,varargin{:});
            dataRate = 60;
            % assume all 2p behaviour data is in 2p_microscope_data folder
            if contains(dataPath{1}, '2p_microscope')
                disp('Analyzing 2p behaviour data...very bold of you...');
                % make more robust for darRig imaging/behaviour experiments by
                % make interleaveEpoch = # probe epochs + 1
                nonProbeEpochs = find([params{1,1}.ordertype]==4);
                interleaveEpoch = nonProbeEpochs(1);
            else
                % original interleaveEpoch for behaviour rigs (no probe)
                interleaveEpoch = 1;
                % added by Joe and Natalia to work with Omer's stimuli
                %nonProbeEpochs = find(~([params{1,1}.ordertype]==5)); % If you
                %are using a different order type, you might need to change
                %this line for it to work! uncomment it
            end
            
            dataType = 'behavioralData';
        elseif imagingData
            disp('Analyzing imaging data...');
            % read in imaging data by hacking Emilio's output
            [flyResp,epochs,params,stim,flyIds,selectedEpochs,dataRate,interleaveEpoch,argumentOut] = ReadImagingData(dataPath,varargin{:});
            dataType = 'imagingData';
            numTotalFlies = size(flyResp,2);
            rigIds = ones(numTotalFlies,1);
        elseif ephysData
            disp('Analyzing e-phys data...');
            [flyResp,epochs,params,stim,flyIds,dataRate] = ReadEphysData(fullDirectory,varargin{:});
            interleaveEpoch = 1;
            dataType = 'ephysData';
            numTotalFlies = size(flyResp,2);
        end
        
        %% save all analysis inputs to D
        D.inputs.flyResp = flyResp;
        D.inputs.epochs = epochs;
        D.inputs.params = params;
        D.inputs.stim = stim;
        D.inputs.flyIds = flyIds;
        D.inputs.dataRate = dataRate;
        D.inputs.numTotalFlies = numTotalFlies;
        D.inputs.dataType = dataType;
        D.inputs.interleaveEpoch = interleaveEpoch;
        D.inputs.getUniqueFliesFlag = getUniqueFliesFlag;
        D.inputs.argumentOut = argumentOut;
    else
        % get all your inputs from D
        flyResp = D.inputs.flyResp;
        epochs = D.inputs.epochs;
        params = D.inputs.params;
        stim = D.inputs.stim;
        flyIds = D.inputs.flyIds;
        dataRate = D.inputs.dataRate;
        numTotalFlies = D.inputs.numTotalFlies;
        dataType = D.inputs.dataType;
        interleaveEpoch = D.inputs.interleaveEpoch;
        getUniqueFliesFlag = D.inputs.getUniqueFliesFlag;
        argumentOut = D.inputs.argumentOut;
    end
    
    if isempty(flyResp)
        return
    end
    
    % Merge data if the same experiment was repeated multiple times on
    % the same fly
    
    if getUniqueFliesFlag
        [flyResp,stim,epochs,params,numTotalFlies,argumentOut{2:2:end}] = GetUniqueFlies(flyResp,stim,epochs,flyIds,params,numTotalFlies,argumentOut{2:2:end});
        flyIdsSel = unique(flyIds, 'stable');
    else
        flyIdsSel = flyIds;
    end
        
    %% Get the path to the analysisFile with GUI if it was not explicitly provided
    if isempty(analysisFile)
        rootFolder = fileparts(which('RunStimulus'));
        
        analysisFile =  UiPickFiles('FilterSpec',fullfile(rootFolder,'analysis','analysisFiles'),'Prompt','Select parameter files to run');
        
        if isempty(analysisFile)
            error('no analysis file chosen');
        end
    end
    
    if ~iscell(analysisFile)
        analysisFile = {analysisFile};
    end

    % Loop through variations of data to run the same analysis on
    % (flyResp can be a cell array with multiple cells when multiple ROI
    % selection fucntions were used (e.g. selecting T4 and T5 from a 
    % single recording using different criteria etc.))
    % I don't know if this happens in behavior or not
    
    a=tic;
    
    for dd = 1:size(flyResp,1)
        % Loop through analysis files and run them
        for aa = 1:length(analysisFile)
            
            % Convert analysis function name string to a function that we
            % can call
            analysisFunction = str2func(analysisFile{aa});
            
            % Prepare input arguments for analysis functions
            argumentInAnalysis = argumentOut;
            
            % Create a flag for non-responsive flies
            % (These quality control happens inside the read data functions)
            nonResponsiveFlies = cellfun('isempty', flyResp(dd, :));
            
            % Remove data and metadata for bad flies
            flyRespAnalysis = flyResp(dd, ~nonResponsiveFlies);
            epochsAnalysis = epochs(dd, ~nonResponsiveFlies);
            paramsAnalysis = params(dd, ~nonResponsiveFlies);
            stimAnalysis = stim(dd, ~nonResponsiveFlies);
            argumentInAnalysis(2:2:end) = cellfun(@(val) val(dd, ~nonResponsiveFlies), argumentOut(2:2:end), 'UniformOutput',false);
            fprintf('The following flies did not respond: %s\n', num2str(find(nonResponsiveFlies)));
            
            % Run the analysis function
            if ~isempty(flyRespAnalysis)
               D.analysis{aa,dd} = analysisFunction(flyRespAnalysis,epochsAnalysis,paramsAnalysis,stimAnalysis,dataRate,dataType,interleaveEpoch,'numTotalFlies',numTotalFlies,varargin{:},argumentInAnalysis{:},'rigIds',rigIds, 'iteration', dd, 'flyIds', flyIds, 'flyIdsSel', flyIdsSel);
            else
                D.analysis{aa,dd} = [];
            end
            fliesUsed = flyIdsSel(~nonResponsiveFlies);
            D.analysis{aa, dd}.fliesUsed = fliesUsed;
        end
    end
    disp(['analysis file took ' num2str(toc(a)) ' seconds to run']);
    
    % Important script that must be run for analysis to succeed
    % fourierGAL4SplineAnalysis;
end
