function respMat = SnipMatToMatrix(snipMat)
    % tries to convert snipMat of form {flies} {epochs,ROIs}
    % [time,trials,TW] to a matrix with diminsions
    % [time,snips,epochs,ROIs,flies,TW] consider using squish afterwards for
    % easy plotting
    permutedInner = cellfun(@(x)cellfun(@(y)permute(y,[1 2 4 5 6 3]),x,'UniformOutput',0),snipMat,'UniformOutput',0);
    permutedOuter = cellfun(@(x)permute(x,[3 4 1 2]),permutedInner,'UniformOutput',0);
    respMat = cell2mat(cat(5,permutedOuter{:}));
end