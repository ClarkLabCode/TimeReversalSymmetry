function [texStr,stimData] = Glider(Q)

% this is to provide the general structure for the texture generating codes
% to be used with PTB in this framework. 

% NOTE: when you create a new stimulus function, you must update the
% stimlookup table in the folder paramfiles. paramfiles will also hold the
% text file giving lists of parameters that comprise an experiment

%when choosing noise values for the sine wave make sure that:
%noiseContrast <= (1-mlum*(contrast+1))/(3*mlum)
%this insures that 3 std of the noise keeps you below a luminence of 1

sii = Q.stims.currStimNum;
p = Q.stims.currParam; % this is what we've got to work with in terms of parameters for this stimulus
f = Q.timing.framenumber - Q.timing.framelastchange; % relative frame number
stimData = Q.stims.stimData;
floc = Q.flyloc; % could potentially use this to update the stimulus as well

texStr.opts = 'full'; % or 'rightleft','rightleftfront', etc. see drawTexture for deets
texStr.dim = 2; % or 2
texStr.scale = [1 1 1]; % using the different lengthscales appropriately.

%% input parameters
whichGlid = p.whichGlid;
framesPerUp = p.framesPerUp;
updateRate = p.updateRate;
    lifespan = 60 * framesPerUp / updateRate;
    % flips per stimulus update
pixX = p.pixX;
btmpWd = round(360/pixX);
diagDirec = p.diagDirec; % one is right
pol = p.pol;
if isfield(p,'reseed')
    reseed = p.reseed;
else
    reseed = 0;
end
contrast = p.contrast;
mLum = p.mLum;
if whichGlid > 10
    varDt = p.varDt;
else
    varDt = [];
end

%% Parameters associated with each specific glider type

[ dx1,dx2,dt1,dt2,glidOrder ] = GlidDisp( whichGlid,varDt );

maxDispX = max([dx1 dx2]);
maxDispT = max([dt1 dt2]);

%% Setup for first frame - needs to come after glider type specifications
if f == 0
    if reseed
        rng(Q.timing.framenumber);
    end
    stimData.age = 0;
    stimData.map = 2*(randi(2,[maxDispT+1,btmpWd + maxDispX])-1.5);
end

%% Draw the gliders

for q = 1:framesPerUp 
    
    if stimData.age == 0  
        % roll back old map
        stimData.map(2:end,:) = stimData.map(1:end-1,:);

        % generate current output in first column of map
        for x = 1:maxDispX
            stimData.map(1,x) = 2*(randi(2)-1.5);
        end
        
        if glidOrder == 2
            for x=maxDispX+1:btmpWd;  
                stimData.map(1,x) = pol * stimData.map(dt1+1,x-dx1); 
                % adding 1 to dt because first row of map is "t=0" - the
                % map that will be printed
            end 
        elseif glidOrder ==3
            for x=maxDispX+1:btmpWd;  
                stimData.map(1,x) = pol * stimData.map(dt1+1,x-dx1)*stimData.map(dt2+1,x-dx2); 
            end            
        end    
        
    end   
    
    if diagDirec == -1
        printOut = fliplr(mLum * (1 + contrast*stimData.map(1,:)));
    else
        printOut = mLum * (1 + contrast*stimData.map(1,:));
    end    
    
    bitMap(:,:,q) = printOut;
    
    % update saved parameters
    stimData.age = mod(stimData.age+1,lifespan);     
end

%always include this line in a stim function to make the texture from the
%bitmap
texStr.tex = CreateTexture(bitMap,Q);
end

