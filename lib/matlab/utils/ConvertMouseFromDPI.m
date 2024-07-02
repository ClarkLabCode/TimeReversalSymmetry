function respAve = ConvertMouseFromDPI(respXY,mouseReads,rollSize)
    % Convert timeseries response from dots read per frame to deg/s and mm/s
    
    mouseReadTime = 0.0026; % amount of time the arudino takes to read all 5 mice
    
    % mouseReadTime is the amount of time the ardunio takes to read all
    % mice. This puts the measurements from dots/matlabRead to dots/second
    normResp = respXY./repmat(mouseReads*mouseReadTime,[1 1 2]);
    
    % divide by 360 to get dots per degree 1000*1/4*pi/360
    normResp(:,:,1) = normResp(:,:,1)/(1000*1/4*pi/360);
    % change dY from dots/second to mm/second
    % 1000 dots per inch .03937 inches per mm flip to get mm per dot
    normResp(:,:,2) = normResp(:,:,2)/(1000*.03937);
    
    % perform rolling average set size to 1 for no average. Recommended at 
    % least 2 because the mouse reads alternate betwen 5 and 4 reads per query
    if(nargin < 3)
        rollSize = 2;
    end
    rollFun = ones(rollSize,1)/rollSize;
    respAve = filter(rollFun,1,normResp);
    
end
    %TODO: Get rid of nan's?
    %Get my own mouse read time