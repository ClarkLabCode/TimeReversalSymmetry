function [epochXtPlot,frameRate]=AnalyzeXtPlot(dataPath,computeCorr,plotFigs)    
    %reads in the data and analyzes the xtPlot

    %% deal with inputs
    %initialize vars these can be changed through varargin with the form
    %func(...'varName','value')
    switch nargin
        case 0
            dataPath = [];
            computeCorr = false;
            plotFigs = true;
        case 1
            computeCorr = false;
            plotFigs = true;
        case 2
            plotFigs = true;
    end
    
    if isempty(dataPath)
        [xtPlot,params] = GetXtPlot();
    else
        [xtPlot,params] = GetXtPlot(dataPath);
    end
    
    % get the frame rate of the projector
    framesPerUp = params(1).framesPerUp;
    frameRate = 60*framesPerUp;
    tRes = 1/(frameRate);
    
    % get the epoch tag for each time point
    epoch = xtPlot(:,3);

    % convert the bit levels in the xt plot to contrast levels
    numLevels = 2^(8/(framesPerUp/3));
    bitsToContrast = round(xtPlot(:,4:end).*(numLevels-1));
    bitsToContrast = (bitsToContrast-numLevels/2)/(numLevels/2);

    % make a vector with the id of where each epoch changes
    epochEnds = [1; find(diff(epoch)); size(epoch,1)];
    numEpochs = size(epochEnds,1)-1;
    epochXtPlot = cell(numEpochs,1);
    correlationPerEpoch = cell(numEpochs,1);

    epochIdNum = zeros(numEpochs,1); 

    for nn = 1:numEpochs
        % extract the epoch xtPlot
        epochXtPlotIn = bitsToContrast((epochEnds(nn)+1):epochEnds(nn+1),:);
        epochIdNum(nn,1) = epoch(epochEnds(nn)+1);

        % If the xtPlots have different numbers of pixels in x, then there
        % will be nan values. Remove thses
        
        nanNanPixels = ~isnan(sum(epochXtPlotIn,1));
        epochXtPlot{nn} = epochXtPlotIn(:,nanNanPixels);
    end

    if plotFigs
        for ee = 1:numEpochs
            pixelSize = 360/(size(epochXtPlot{ee},2));
            tickX = (0:pixelSize:360-pixelSize)';
            tickY = (0:tRes:tRes*(size(epochXtPlot{ee},1)-1))'*1000;
            
            figH=MakeFigure;
            figTitle = ['xtPlot epoch ' num2str(ee)];
            set(figH,'name',figTitle,'NumberTitle','off');
            title(figTitle);
            imagesc(tickX,tickY,epochXtPlot{ee});
            ConfAxis('tickX',tickX,'tickLabelX',tickX,'tickY',tickY,'tickLabelY',round(tickY*10)/10,'labelX',['space (' char(186) ')'],'labelY','time (ms)');
            colormap(gray);
        end
    end

    if computeCorr
        for nn = 1:numEpochs
            % perform the cross correlation in fourier space. I tried to do
            % a circular convulation with imfilter and got artifacts for
            % some reason
            correlationPerEpoch{nn} = fftshift(ifft2(fft2(epochXtPlot{nn}).*conj(fft2(epochXtPlot{nn}))))/numel(epochXtPlot{nn})/var(epochXtPlot{nn}(:));

            if plotFigs
                pixelSize = 360/size(epochXtPlot{nn},2);
                tickX = (-180:pixelSize:180-pixelSize)';
                tDuration = tRes*size(epochXtPlot{nn},1);
                tickY = (-tDuration/2:tRes:tDuration/2-tRes)'*1000;
            
                figH=MakeFigure;
                figTitle = ['autocorrelation epoch ' num2str(nn)];
                set(figH,'name',figTitle,'NumberTitle','off');
                title(figTitle);
                imagesc(tickX,tickY,correlationPerEpoch{nn});
                ConfAxis('tickX',tickX,'tickLabelX',tickX,'tickY',tickY,'tickLabelY',round(tickY*10)/10,'labelX',['dx (' char(186) ')'],'labelY','dt (ms)');
                colormap(gray);
            end
        end
    end
end