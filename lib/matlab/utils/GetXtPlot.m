function [xtPlot,params] = GetXtPlot(dataPathIn)
    if nargin<1
        dataPathIn = [];
    end
    
    sysConfig = GetSystemConfiguration();
    dataLocation = sysConfig.dataPath;

    if isempty(dataPathIn)
        dataPathIn = UiPickFiles('FilterSpec',fullfile(dataLocation,'xtplot'),'Prompt','Choose folders containing the files to be analyzed');
        dataPath = dataPathIn{1};
    else
        dataPath = fullfile(dataLocation,dataPathIn);
    end
    
    xtPlotsInSubDir = DirRec(dataPath,'xtPlot*')';
    paramsInSubDir = DirRec(dataPath,'chosenparams*')';

    xtPlot = importdata(xtPlotsInSubDir{1});
    params = load(paramsInSubDir{1});
    params = params.params;
end