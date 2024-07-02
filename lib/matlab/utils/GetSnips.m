function snipMat = GetSnips(resp,epochStartTimes,durations)
    % Converts response from [time flies (dx,dy)] format to a cell array of
    % [epochs flies] where each cell has a matrix of format [time trial
    % (dx,dy).
    
    if size(resp,2) ~= size(epochStartTimes,2)
        error('number of ROIs must be consistent');
    end
    
    %% To calculate the end time for each epoch, we will add the epoch's
    % duration to the start time.
    [numEpochs, numFlies] = size(epochStartTimes);
    [flyIndexMesh,epochIndexMesh] = meshgrid(1:numFlies,1:numEpochs);
    % Put the durations in the same layout as the cell level of
    % epochStartTimes so that we can call cellfun on them.
    epochDurations = num2cell(reshape(durations(epochIndexMesh),size(epochIndexMesh)));
    % Adds the duration to the start times
    epochEndTimes = cellfun(@(start,duration)start+duration-1,epochStartTimes,epochDurations,'UniformOutput',false);
    
    %% From the start times and end times, we can get the full set of frames
    % to include in our final snipmat
    epochRanges = cellfun(@(startTimes,endTimes) ... %Take in matricies that define the start and end times
                              ... % For each element of these matricies, create a range from the start to end times with the colon function
                              cell2mat(arrayfun(@colon,startTimes,endTimes,'UniformOutput',false))' ...
                          ,epochStartTimes,epochEndTimes,'UniformOutput',false);
    
    %% For each epoch and fly, take in the frames to grab, look them up in
    % resp and then reshape to put in [time trials (turn walk)] format
    snipMat = cellfun(@(frames,duration,fly)reshape(resp(frames,fly,:),duration,[],size(resp,3)),...
                      epochRanges,epochDurations,num2cell(flyIndexMesh),'UniformOutput',false);
end