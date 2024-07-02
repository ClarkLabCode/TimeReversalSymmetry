function meanSubtracted = MeanSubtractTurns(snipMat,turnMeans)

    % move through the cell array and subtract off the turn means from each
    % fly and each epoch
    
    meanSubtracted = cell(size(snipMat));
    
    for ff = 1:size(snipMat,2)
        for ee = 1:size(snipMat,1)
            meanSubtracted{ee,ff} = snipMat{ee,ff};
            meanSubtracted{ee,ff}(:,:,1) = meanSubtracted{ee,ff}(:,:,1)-turnMeans(ff);
        end
    end
end