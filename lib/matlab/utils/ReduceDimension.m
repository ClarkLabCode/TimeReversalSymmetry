function outSnipMat = ReduceDimension(inSnipMat,dimension,func)
% Take the mean along a given dimension.
% Takes in a snipMat and a string specifying the dimension to average
% across: 'flies','epochs','time', or 'trials'. Optionally take in a
% function handle to apply to the dimension. 
% the function func to perform this reduction should have inputs of
% func(data,dimension). dimension can also be a cell
% array of dimensions to apply in order.

    if nargin < 3
        func = @nanmean;
%         func = @nansum;
    elseif isempty(func)
        % Allows for calling the function with a variable func, including
        % one that doesn't change the input at all
        outSnipMat = inSnipMat;
        return
    end
    
    if iscell(dimension) % If given a cell array of dimensions, apply each
                         % in sequential order by calling ReduceDimension
       tempSnipMat = inSnipMat;
       for ii = 1:length(dimension)
           tempSnipMat = ReduceDimension(tempSnipMat,dimension{ii},func);
       end
       outSnipMat = tempSnipMat;
    else % If given a single dimension, apply function on that dimension
        
        [numEpochs,numROIs] = size(inSnipMat);
        switch dimension
            case 'Rois'
                outSnipMat = cell(numEpochs,1);
                for ee = 1:numEpochs
                    %Take all flys in an epoch, stack them in the 4th
                    %dimension, convert to a matrix (implicitly assume 
                    %submatricies are of equal size) then perform func on them
                    %along the fourth dimension
                    outSnipMat{ee,1} = func(cat(4,inSnipMat{ee,:}),4);
                end
            case 'epochs'
                outSnipMat = cell(1,numROIs);
                for rr = 1:numROIs
                    %Take all epochs per fly, stack them in the 4th
                    %dimension, convert to a matrix (implicitly assume 
                    %submatricies are of equal size) then perform func on them
                    %along the fourth dimension
                    outSnipMat{1,rr} = func(cat(4,inSnipMat{:,rr}),4);
                end
            case 'time'
                funcCheck = {@max, @min, @nanmax, @nanmin, @diff};
                goodFunc = ~any(cellfun(@(fIn, fCheck) isequal(fIn, fCheck), repmat({func}, 1, length(funcCheck)), funcCheck));
                if goodFunc
                    outSnipMat = cellfun(@(x)func(x,1),inSnipMat,'UniformOutput',false);
                else
                    outSnipMat = cellfun(@(x)func(x,[], 1),inSnipMat,'UniformOutput',false);
                end
            case 'trials'
                funcCheck = {@max, @min, @nanmax, @nanmin, @diff};
                goodFunc = ~any(cellfun(@(fIn, fCheck) isequal(fIn, fCheck), repmat({func}, 1, length(funcCheck)), funcCheck));
                if goodFunc
                    outSnipMat = cellfun(@(x)func(x,2),inSnipMat,'UniformOutput',false);
                else
                    outSnipMat = cellfun(@(x)func(x,[], 2),inSnipMat,'UniformOutput',false);
                end
            case 'flies'
                
                % Get rid of any flies that had no response, if they happen
                % to exist
%                 inSnipMat(cellfun('isempty', inSnipMat)) = [];
                numFlies = length(inSnipMat);
                [numEpochs,numROIs] = size(inSnipMat{1});
                
                outSnipMat = cell(1);
                outSnipMat{1} = cell(numEpochs,numROIs);
                
                for ee = 1:numEpochs
                    for rr = 1:numROIs
                        [numTime,numTrials,numTW] = size(inSnipMat{1}{ee,rr});
                        
                        outSnipMat{1}{ee,rr} = zeros(numTime,numTrials,numTW,numFlies);
                        
                        for ff = 1:numFlies
                            outSnipMat{1}{ee,rr}(:,:,:,ff) = inSnipMat{ff}{ee,rr};
                        end
                        
                        outSnipMat{1}{ee,rr} = func(outSnipMat{1}{ee,rr},4);
                    end
                end
        end
    end
end