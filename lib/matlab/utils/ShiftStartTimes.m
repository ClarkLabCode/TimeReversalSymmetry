function output = ShiftStartTimes(startTimes,snipShift)
    %Takes in a snipMat format cell array of start times and shifts all
    %values by snipShift
    
    output = cellfun(@(x)x+snipShift, startTimes,'UniformOutput',false);
    
end