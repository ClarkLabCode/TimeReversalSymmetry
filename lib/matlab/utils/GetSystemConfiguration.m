function sysConfig = GetSystemConfiguration()
    rootFolder = fileparts(which('RunStimulus'));

    defaultConfig = LoadSettings(fullfile(rootFolder,'defaultConfig.csv'));
    try
        thisCompConfig = LoadSettings(fullfile(rootFolder,'sysConfig.csv'));
    catch
        thisCompConfig = struct;
    end
    sysConfig = CatStruct(defaultConfig,thisCompConfig);
    
    fields = fieldnames(sysConfig);
    
    for fn = 1:length(fields)
        if ~isempty(str2num(sysConfig.(fields{fn})))
            sysConfig.(fields{fn}) = str2num(sysConfig.(fields{fn}));
        end
    end
end