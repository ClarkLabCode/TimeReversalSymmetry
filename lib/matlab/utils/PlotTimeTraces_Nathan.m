function analysis = PlotTimeTraces_Nathan(flyResp,epochs,params,stim,dataRate,dataType,interleaveEpoch,varargin)
    combOpp = 1; % logical for combining symmetic epochs such as left and right
    numIgnore = interleaveEpoch; % number of epochs to ignore
    figLeg = {};
    ttDuration = [];
    ttSnipShift = -500;
    imagingSelectedEpochs = {'' ''};
    fTitle = '';
    plotOnly = '';
    reassignEpochs = '';
    plotFigs = 1;
    color = [];
    numFlies = [];


    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end
    
    if ~iscell(imagingSelectedEpochs)
        imagingSelectedEpochs = num2cell(imagingSelectedEpochs);
    end
    
    % Convert variables related to time from units of milliseconds to
    % samples
    
    if ~isempty(reassignEpochs) % resize params struct array to get rid of
                                % merged epochs and sum the durations of
                                % the merged epochs
        [newParamList,oldParamIdx,~] = unique(reassignEpochs);
        newParams(newParamList) = params{1}(oldParamIdx);
        for param = newParamList
            newParams(param).duration = sum(cell2mat({params{1}(reassignEpochs == param).duration}));
        end
        params = newParams;
    end
    
    %% duration and snip shift should be entered in miliseconds

    longestDuration = params{1}(interleaveEpoch+1).duration*1000/60;
    for pp = interleaveEpoch+1:length(params{1})
        thisDuration = params{1}(pp).duration*1000/60;
        if thisDuration>longestDuration
            longestDuration = thisDuration;
        end
    end
    
    % snip shift reads in ttSnipShift so that PlotTimeTraces and CombAndSep
    % do not interact
    snipShift = ttSnipShift;
    
    if isempty(ttDuration)
        duration = longestDuration + 2500;
    else
        duration = ttDuration;
    end
    
    if isempty(numFlies)
        numFlies = length(flyResp);
    end
    averagedRois = cell(1,numFlies);
    
    %% get processed trials
    
    for ff = 1:numFlies
%         flyResp{ff}(:,1,2) = -180/pi*stim{ff}(:,2);

        if ~isempty(reassignEpochs)
            for ii = 1:length(reassignEpochs)
                newEpochs(epochs{ff} == ii) = reassignEpochs(ii);
            end
            epochs{ff} = newEpochs';
        end
        
        analysis.indFly{ff} = GetProcessedTrials(flyResp{ff},epochs{ff},params{1},dataRate,...
                                                 dataType,varargin{:},'duration',duration, ...
                                                 'snipShift',snipShift);
                                      
        % Remove ignored epochs
        selectedEpochs = analysis.indFly{ff}{end}.snipMat(numIgnore+1:end,:);

        % write to output structure
        analysis.indFly{ff}{end+1}.name = 'selectedEpochs';
        analysis.indFly{ff}{end}.snipMat = selectedEpochs;

        %% average over trials
        averagedTrials = ReduceDimension(selectedEpochs,'trials');

        % write to output structure
        analysis.indFly{ff}{end+1}.name = 'averagedTrials';
        analysis.indFly{ff}{end}.snipMat = averagedTrials;

        %% combine left ward and rightward epochs
        if combOpp
            combinedOpposites = CombineOpposites_Nathan(averagedTrials, varargin{:});
        else
            combinedOpposites = averagedTrials;
        end

        % write to output structure
        analysis.indFly{ff}{end+1}.name = 'combinedOpposites';
        analysis.indFly{ff}{end}.snipMat = combinedOpposites;

        %% average over Rois (or over behaving flies I think -BA)
        averagedRois{ff} = ReduceDimension(combinedOpposites,'Rois');

        % write to output structure
        analysis.indFly{ff}{end+1}.name = 'averagedRois';
        analysis.indFly{ff}{end}.snipMat = averagedRois{ff};


        %% Change names of analysis structures
        analysis.indFly{ff} = MakeAnalysisReadable(analysis.indFly{ff});
    end
    
    %% convert from snipMat to matrix wtih averaged flies
    averagedFlies = ReduceDimension(averagedRois,'flies',@nanmean);
    averagedFliesSem = ReduceDimension(averagedRois,'flies',@NanSem);
    
    respMat = SnipMatToMatrix(averagedFlies); % turn snipMat into a matrix
    respMatPlot = permute(respMat,[1 3 6 7 2 4 5]);

    respMatSem = SnipMatToMatrix(averagedFliesSem); % turn snipMat into a matrix
    respMatSemPlot = permute(respMatSem,[1 3 6 7 2 4 5]);
    
    analysis.respMatPlot = respMatPlot;
    analysis.respMatSemPlot = respMatSemPlot;
    analysis.plotFigs = plotFigs;
        
    
    %%
    if isempty(figLeg) && isfield(params{1},'epochName')
        for ii = (1+numIgnore):length(params{1})
            if ischar(params{1}(ii).epochName)
                figLeg{ii-numIgnore} = params{1}(ii).epochName;
            else
                figLeg{ii-numIgnore} = '';
            end
        end
    end
            
    timeX = ((1:round(duration*dataRate/1000))'+round(snipShift*dataRate/1000))*1000/dataRate;
    analysis.timeX = timeX;

    %% plot
    if plotFigs
        middleTime = linspace(0,longestDuration,5);
        timeStep = middleTime(2)-middleTime(1);
        earlyTime = fliplr(0:-timeStep:snipShift);
        endTime = longestDuration:timeStep:duration+snipShift;
        plotTime = round([earlyTime(1:end-1) middleTime endTime(2:end)]*10)/10;

        switch dataType
            case 'behavioralData'
                yAxis = {['turning response (' char(186) '/s)'],'walking response (fold change)'};
            case 'imagingData'
                yAxis = {'\DeltaF / F'};
            case 'ephysData'
                yAxis = {'Neural Response (mV)'};
        end

        if strcmp(dataType,'imagingData')
            finalTitle = [fTitle ': ' imagingSelectedEpochs{1} ' - ' imagingSelectedEpochs{2}];
        else
            finalTitle = fTitle;
        end

        for pp = 1:size(respMatPlot,3)
            if strcmp(plotOnly,'walking') && pp == 1
                continue;
            end
            if strcmp(plotOnly,'turning') && pp == 2
                continue;
            end
            if strcmp(plotOnly,'none')
                continue;
            end
            MakeFigure;
            % keyboard;
            PlotXvsY(timeX,respMatPlot(:,:,pp),'error',respMatSemPlot(:,:,pp),'plotColor',color);
            hold on;
            PlotConstLine(0);
            PlotConstLine(0,2);
            PlotConstLine(longestDuration,2);
            
            if pp == 2
                PlotConstLine(1);
            end

            %ConfAxis('tickX',plotTime,'tickLabelX',plotTime,'labelX','time (ms)','labelY',[yAxis{pp} ' - ' num2str(numFlies) '/' num2str(numTotalFlies) ' flies'],'fTitle',finalTitle,'figLeg',figLeg);
            hold off;
        end
    end
end