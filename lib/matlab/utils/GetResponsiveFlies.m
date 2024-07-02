function [responsive,numTotalFlies] = GetResponsiveFlies(resp,epochs,flyIds)

    absCutOffWalk = 1; % mm/sec
    absCutOffTurn = 40; % deg/sec
    relCutOffTurn = 1.5; % std/turn
    deadFlyCutoff = 0.9;
    
    numFlies = size(epochs,2);
    interelaveEpoch = epochs==1;
    interleaveEpochInd = find(diff([zeros(1,numFlies); interelaveEpoch])==-1);
    
    [row,col] = ind2sub(size(epochs),interleaveEpochInd);
    walkingMean = zeros(1,numFlies);
    turningMean = zeros(1,numFlies);
    turningStd = zeros(1,numFlies);
    numTrials = zeros(1,numFlies);

    numTotalFlies = size(resp,2);
    fractionTimeStopped = 1-mean(resp(:,:,2)>0,1);
    deadFlies = fractionTimeStopped>deadFlyCutoff;
    numTotalFlies = numTotalFlies-sum(deadFlies);
    
    if ~isempty(interleaveEpochInd)
        % duration to average over in frames;
        interleaveAverageDuration = min([60 row(1)]);

        for ii = 1:length(col)
            walkingMean(1,col(ii)) = walkingMean(1,col(ii))+mean(resp((row(ii)-interleaveAverageDuration+1):row(ii),col(ii),2));
            turningMean(1,col(ii)) = turningMean(1,col(ii))+mean(resp((row(ii)-interleaveAverageDuration+1):row(ii),col(ii),1));
            turningStd(1,col(ii)) = turningStd(1,col(ii))+std(resp((row(ii)-interleaveAverageDuration+1):row(ii),col(ii),1));
            numTrials(1,col(ii)) = numTrials(1,col(ii)) + 1;
        end

        walkingMean = walkingMean./numTrials;
        turningMean = turningMean./numTrials;
        turningStd = turningStd./numTrials;
    else
        walkingMean = mean(resp(:,:,2),1);
        turningMean = mean(resp(:,:,1),1);
        turningStd = std(resp(:,:,1),[],1);
    end
    
    % flies that don't have a high enough std
    fliesHighTurnSTD = turningStd>absCutOffTurn;

    % flies that don't walk fast enough
    fliesHighWalk = walkingMean>absCutOffWalk;

    % don't have a high enough std to mean ratio
    fliesHighSTDtoMean = turningStd./abs(turningMean)>relCutOffTurn;
    
    responsive = logical(fliesHighTurnSTD & fliesHighWalk & fliesHighSTDtoMean);
    
    % get rid of duplicated flies in our total flies count
    uniqueFlyIds = unique(flyIds);
    aliveRemovedFlies = (~responsive & ~deadFlies);
    for uu = 1:length(uniqueFlyIds)
        theseFlies = flyIds==uniqueFlyIds(uu);
        theseFliesRemoved = aliveRemovedFlies(theseFlies);
        numTotalFlies = numTotalFlies - sum(theseFliesRemoved);
        if all(theseFliesRemoved)
            numTotalFlies = numTotalFlies + 1;
        end
    end
end