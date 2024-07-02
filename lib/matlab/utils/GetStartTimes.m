function epochStartTimes = GetStartTimes(epochs)
    
    % Find the frames where the epoch changes. Note that the first epoch
    % won't be counted.
    [epochChangeFrames, epochChangeFlies] = find(diff(epochs));
    % Account for shift caused by the diff function
    epochChangeFrames = epochChangeFrames+1; 
    
    % Get linear index of epochs that are starting at the change point
    epochIdx = sub2ind(size(epochs),epochChangeFrames, epochChangeFlies);
    
    % Make a snipMat-like data format for the start times of the epochs.
    % Accumarray is given the {} function which concatonates all the change
    % frames for a given epoch and fly.
    epochStartTimes = accumarray([epochs(epochIdx) epochChangeFlies], epochChangeFrames, [], @(x){x});