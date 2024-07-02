function output = NormalizeWalking(inputSnipMat,normValues,epsilon)
    % Takes in a response snipMat and a snipMat with time and trials
    % averaged. Output is a snipMat normalized by the averaged values.
    % Optional epsilon can be provided to avoid dividing by 0.
    
    if (nargin < 3)
        epsilon = 0;
    end
    
    output = cell(size(inputSnipMat));
    
    [numEpochs,numRois] = size(inputSnipMat);
    for epoch = 1:numEpochs
        for rois = 1:numRois
            thisResponse = inputSnipMat{epoch,rois};
            [numFrames, numTrials,~] = size(thisResponse); 
            thisNormValues = repmat(normValues{1,rois}(1,1,2),[numFrames numTrials]);
            output{epoch,rois}(:,:,1) = thisResponse(:,:,1);
            output{epoch,rois}(:,:,2) = thisResponse(:,:,2)./(thisNormValues+epsilon);
        end
    end
end