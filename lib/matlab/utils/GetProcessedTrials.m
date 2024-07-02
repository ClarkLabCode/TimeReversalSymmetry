function analysis = GetProcessedTrials(flyResp,epochs,params,dataRate,dataType,varargin)
    normWalkDuration = 500; % number of frames in the interleave before the epoch to normalize walking to
    normWalkShift = -500;
    epsilon = 0.001; % when normalizing walking, may want to add epsilon to the denominator to avoid errors
    analysis = cell(0,1);
    duration = [];
    normTurnDuration = 500;
    normTurnShift = -500;
    timeLimits = [];
    linescan = false;
    
    switch dataType
        case 'behavioralData'
            normWalk = 1;
            meanSubtract = 1;
            subtractInterleave = 0;
            snipShift = 100; % behavioral delay is ~80-100ms
        case 'imagingData'
            normWalk = 0;
            meanSubtract = 0;
            subtractInterleave = 0;
            snipShift = 0;
        case 'ephysData'
            normWalk = 0;
            meanSubtract = 0;
            subtractInterleave = 1;
            snipShift = 0;
    end
    
    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end
    if linescan
        dataRate = dataRate * 1024; % 1024 is the number of lines per image
    end
    % Convert variables related to time from units of milliseconds to
    % samples
    normWalkDuration = round(normWalkDuration*dataRate/1000)*normWalk;
    normWalkShift = round(normWalkShift*dataRate/1000)*normWalk;
    normTurnDuration = round(normTurnDuration*dataRate/1000)*subtractInterleave;
    normTurnShift = round(normTurnShift*dataRate/1000)*subtractInterleave;
    duration = round(duration*dataRate/1000);
    snipShift = round(snipShift*dataRate/1000);
    timeLimits = round(timeLimits*dataRate/1000);
    
    %% This file will
    % GetSnips
    % NormalizeWalking
    % MeanSubtractTurns
    
    %% Limit analysis to the selected time period
    if ~isempty(timeLimits)
        flyResp = flyResp(1+timeLimits(1):timeLimits(2),:);
        epochs = epochs(1+timeLimits(1):timeLimits(2),:);
    end
    
    %% get snips of time over duration of epoch
    % find the index at which each epoch starts
    epochStartTimes = GetStartTimes(epochs);
    % if by bad chance the last epoch had no trials this will return the
    % wrong size cell array. fix here
    for ll = (size(epochStartTimes,1)+1):length(params)
        epochStartTimes{ll} = [];
    end
    
    if isempty(duration)
        % Divide by 60 for frames/s display from the projector
        epochDurations = round([params.duration]'/60*dataRate);
    else
        epochDurations = zeros(length(params),1)+duration;
    end
    
    %%%%%% this assumes that all epoch start times are the same for each
    %%%%%% ROI. Needs to be fixed so that FilterStartTimes can take in an
    %%%%%% array of max ranges not just a single set value
    minShift = min([snipShift normWalkShift normTurnShift+snipShift]);
    maxShift = max([snipShift normWalkShift normTurnShift+snipShift]);
    
    maxRange = [minShift maxShift];
    
    % get rid of start times and are too early and end times that are too late
    filteredStartTimes = FilterStartTimes(epochStartTimes,maxRange,epochDurations,size(epochs,1));
    
    % shift all start times by a set amount to account for behavioral delay
    shiftedEpochStartTimes = ShiftStartTimes(filteredStartTimes,snipShift);
    % get the snips from the matrix and put them in a snip mat of the
    % format {epochs, flies}[time trials]
    snipMat = GetSnips(flyResp,shiftedEpochStartTimes,epochDurations);
    
    % write to output structure
    analysis{end+1}.name = 'snipMat';
    analysis{end}.snipMat = snipMat;
    analysis{end}.epochList = epochs;
    analysis{end}.flyResp = flyResp;
    analysis{end}.shiftedEpochStartTimes = shiftedEpochStartTimes;
    analysis{end}.epochDurations = epochDurations;
    
    %% normalize walking
    % get the start times for the snip during interleave to normalize walking by
    normWalkStartTimes = ShiftStartTimes(filteredStartTimes,normWalkShift);
    % get the duration of the snip during interleave to normalize walking by
    normWalkDurations = normWalkDuration*ones(length(epochDurations),1);
    % get the snip during interleave to normalize walking by
    walkRespDuringInterleave = GetSnips(flyResp,normWalkStartTimes,normWalkDurations);
    % average the snip during interleave to normalize walking by
    beforeEpochAveWalk = ReduceDimension(walkRespDuringInterleave,{'time' 'trials' 'epochs'});

    % normalize walking
    if normWalk
        normalizedWalking = NormalizeWalking(snipMat,beforeEpochAveWalk,epsilon);
    else
        normalizedWalking = snipMat;
    end

    % write to output structure
    analysis{end+1}.name = 'normalizedWalking';
    analysis{end}.snipMat = normalizedWalking;
    analysis{end}.normWalkStartTimes = normWalkStartTimes;
    analysis{end}.normWalkDurations = normWalkDuration;
    analysis{end}.beforeEpochAveWalk = beforeEpochAveWalk;
    analysis{end}.walkRespDuringInterleave = walkRespDuringInterleave;
    
    %% mean subtract turns
    % get mean of time trace and subtract from turns
    
    if meanSubtract
        turnMeans = nanmean(flyResp(:,:,1),1);
        meanSubtracted = MeanSubtractTurns(normalizedWalking,turnMeans);
    else
        meanSubtracted = normalizedWalking;
    end

    % write to output structure
    analysis{end+1}.name = 'meanSubtracted';
    analysis{end}.snipMat = meanSubtracted;
    
    %% subtract interleave from trace

    % get the start times for the snip during interleave to subtract from
    % the trace
    interSubtractTurnStartTimes = ShiftStartTimes(filteredStartTimes,normTurnShift+snipShift);
    % get the duration of the snip during interleave to normalize walking by
    interSubtractTurnDurations = normTurnDuration*ones(size(epochDurations));
    % get the snip during interleave to normalize walking by
    turnRespDuringInterleave = GetSnips(flyResp,interSubtractTurnStartTimes,interSubtractTurnDurations);
    % average the snip during interleave to normalize walking by
    beforeEpochAveTurn = ReduceDimension(turnRespDuringInterleave,{'time'});

    % normalize walking
    if subtractInterleave
        interSubtractTurning = SubtractInterleaveFromTurning(meanSubtracted,beforeEpochAveTurn);
    else
        interSubtractTurning = meanSubtracted;
    end

    % write to output structure
    analysis{end+1}.name = 'normalizedTurning';
    analysis{end}.snipMat = interSubtractTurning;
    analysis{end}.normTurnStartTimes = interSubtractTurnStartTimes;
    analysis{end}.normTurnDuration = normTurnDuration;
    analysis{end}.turnRespDuringInterleave = turnRespDuringInterleave;
end