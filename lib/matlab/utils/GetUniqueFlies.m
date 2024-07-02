function [flyRespOut,stimOut,epochsOut,paramsOut,numTotalFlies,varargout] = GetUniqueFlies(flyRespIn,stimIn,epochsIn,flyIds,paramsIn,numTotalFlies,varargin)
    flyRespOut = flyRespIn;
    stimOut = stimIn;
    epochsOut = epochsIn;
    varargout = varargin;
    paramsOut = paramsIn;
    selectedFlies = true(1,size(flyRespIn,2));
    behaviorData = size(flyRespOut{1,1},3)==2;
    
    % go through all the fly ids
    for ii = 1:length(flyIds)
        % the matrix of comparisons is symmetric, only go through the upper
        % triangle
        for jj = (ii+1):length(flyIds)
            % if the fly ids of two flies are a match, concatinate their
            % ROIs, stimulus data, and epoch vectors
            if flyIds(ii) == flyIds(jj)
                for dd = 1:size(flyRespOut,1)
                    % Note that we're not doing anything with params
                    % because we assume they're the same one--give a
                    % warning if they're not, though! (Don't error
                    % because a few files don't really depend on the
                    % statistics that this concatenation uses, so we
                    % might still want to analyze disparate parameter
                    % files with those--think PlotTwoPhotonTimeTraces
                    % or EdgeSelectivityAnalysis)
                    if ~isequal(paramsIn{dd,ii}, paramsIn{dd, jj})
                        warning('Something''s probably wrong--did you run one fly with two different parameter files that you''re now trying to analyze together?')
                    end
                        
                    if behaviorData
                        flyRespOut{dd,ii} = cat(1,flyRespOut{dd,ii},flyRespOut{dd,jj});
                        stimOut{dd,ii} = cat(1,stimOut{dd,ii},stimOut{dd,jj});
                        epochsOut{dd,ii} = cat(1,epochsOut{dd,ii},epochsOut{dd,jj});
                    else
                        % sometimes experiments are of different lengths
                        % and I don't know why. At that point this
                        % concatination will fail so I just selected the
                        % shorter of the two and pair them up. This is bad
                        % and could create errors
                        if size(flyRespOut{dd,ii},1) ~= size(epochsOut{dd,ii}, 1)
                            keyboard
                        end
                        
                        minLengthFr = min([size(flyRespOut{dd,ii},1) size(flyRespOut{dd,jj},1)]);
                        flyRespOut{dd,ii} = cat(2,flyRespOut{dd,ii}(1:minLengthFr,:),flyRespOut{dd,jj}(1:minLengthFr,:));
                        epochsOut{dd,ii} = cat(2,epochsOut{dd,ii}(1:minLengthFr,:),epochsOut{dd,jj}(1:minLengthFr,:));
                        % Not really sure why sometimes the stimOuts are
                        % not the same size... but we'll just go with it
                        % here >.> Not used in Emilio's analysis functions?
                        % <.<
                        minLengthSo = min([size(stimOut{dd,ii},1) size(stimOut{dd,jj},1)]);
                        
                        try
                            stimOut{dd,ii} = cat(3,stimOut{dd,ii}(1:minLengthSo,:),stimOut{dd,jj}(1:minLengthSo,:));
                            % seems not to work if you ran the same stim file 3 times on the
                            % same fly, so try this:
                        catch
                            if flyIds(ii) == flyIds(jj) && flyIds(ii) == flyIds(jj-1) && flyIds(ii) == flyIds(jj-2)
                                stimOut{dd,ii} = cat(3,stimIn{dd,jj-2}(1:minLengthSo,:),stimIn{dd,jj-1}(1:minLengthSo,:), stimIn{dd,jj}(1:minLengthSo,:));
                            end
                        end
                        
                    for ll = 1:length(varargout)
                        minLength = min([size(varargout{ll}{dd,ii},1) size(varargout{ll}{dd,jj},1)]);
                        varargout{ll}{dd,ii} = cat(2, varargout{ll}{dd,ii}(1:minLength,:), varargout{ll}{dd,jj}(1:minLength,:));
                    end
                    end
                    selectedFlies(jj) = 0;
                end
                numTotalFlies = numTotalFlies-1;
                disp(['Flies ' num2str(ii) ' and ' num2str(jj) ' were combined']);
            end
        end
    end
    
    % remove the flies that were duplicates
    flyRespOut = flyRespOut(:,selectedFlies);
    stimOut = stimOut(:,selectedFlies);
    epochsOut = epochsOut(:,selectedFlies);
    paramsOut = paramsOut(:, selectedFlies);
    for ll=1:length(varargout)
        % We make an assumption here that chars shouldn't be contatenated,
        % but rather put into different cells
        if ischar(varargout{ll})
            varargout{ll} = varargout{ll}(:,selectedFlies);
        else
            varargout{ll} = varargout{ll}(:,selectedFlies);
        end
    end
end