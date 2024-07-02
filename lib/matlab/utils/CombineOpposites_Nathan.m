function combinedResp = CombineOpposites_Nathan(snipMat, varargin)
    % move through the cell array average walking and 
    % antisymmetrically average turning
    if ~(mod(size(snipMat,1),2) == 0)
        error('to combine opposites you must have an even number of epochs');
    end

    asymAve = 1;
    xFlipPairs = cell(1, size(snipMat, 1)/2);
    for i = 1:size(xFlipPairs, 2)
        xFlipPairs{i} = [i*2-1, i*2];
    end


    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end
    
    combinedResp = cell(size(snipMat,1)/2,size(snipMat,2));

    % there may be faster ways to do this
    

    for ff = 1:size(snipMat,2)
        for ee = 1:size(xFlipPairs,2)
            pair = xFlipPairs{ee};
            combinedResp{ee,ff} = zeros(size(snipMat{ee,ff}));

            if asymAve
                combinedResp{ee,ff}(:,:,1) = nanmean(cat(3,snipMat{pair(1),ff}(:,:,1),-snipMat{pair(2),ff}(:,:,1)),3);
            else
                combinedResp{ee,ff}(:,:,1) = nanmean(cat(3,snipMat{pair(1),ff}(:,:,1),snipMat{pair(2),ff}(:,:,1)),3);
            end
            
            if size(combinedResp{ee,ff},3) == 2
                combinedResp{ee,ff}(:,:,2) = nanmean(cat(3,snipMat{pair(1),ff}(:,:,2),snipMat{pair(2),ff}(:,:,2)),3);
            end
        end
    end
end