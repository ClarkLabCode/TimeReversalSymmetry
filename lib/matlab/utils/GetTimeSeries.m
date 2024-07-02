function [resp,stim,epochs,mouseReads] = GetTimeSeries(respIn,stimIn)
    % input form [time (flies,dx,dy) files] - parentheis indicate variables in the same diminsion
    % output form [time (flies) (dx,dy)]
    
    %% put stim values in form [time stimVal# flies]
    
    % stimIn is in [time value files] format. ceil(1/2n:1/n:end) trick
    % repeats each element n times along the z dimension. TODO: When 
    % everyone has migrated to 2015a or higher, replace with repelem
    % function. Start with 1/2n because with the way floating point math
    % works, ceil(15*1/5) can be 4.
    stim = stimIn(:,4:end,ceil(1/10:1/5:end));
    
    %% put epoch values in form [time flies]
    % Third column of stim is epoch #, the ceil(1/5...) trick repeats
    % each file 5 times to get flies, then squeeze to put flies in second
    % dimension.
    epochs = squeeze(stimIn(:,3,ceil(1/10:1/5:end)));
    
    %% put numReads in form [time flies]
    % numReads found in column 18 of resp. Use average numReads if not
    % available.
    rawReads = squeeze(respIn(:,18,:));
    filesWithoutReads = all(rawReads == 0);
    if(filesWithoutReads)
        warning(['File number(s) ' num2str(filesWithoutReads) ' do(es) not have mouse read counts.\n' ...
                 'Setting read counts to 1']);
        rawReads(:,filesWithoutReads) = 1;
    end
        
    filesWithZeros = any(rawReads == 0);
    if(filesWithZeros)
        warning(['File number(s) ' num2str(filesWithZeros) ' contain(s) zeros in mouse read counts.']);
    end
    
    % Use same trick as above.
    mouseReads = rawReads(:,ceil(1/10:1/5:end));
    
    %% change dxdy data form from [time (5*dx,5*dy) files] to [time flies (dx,dy)]
    
    numTimepoints = size(respIn,1);
    % dx values are columns 3-7, dy values are columns 8-12
    resp = cat(3,reshape(respIn(:,3:7,:),numTimepoints,[]), ...
                 reshape(respIn(:,8:12,:),numTimepoints,[]));


% TODO: support flies/file != 5