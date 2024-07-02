function [settings] = LoadSettings(settingsFile)

    f = fopen(settingsFile);

    settingsCell = textscan(f,'%s %s','Delimiter',',');
    fclose(f);

    assert(all(size(settingsCell{1}) == size(settingsCell{2})), ...
        'Settings File Error: column lengths do not match');

    for ii=1:size(settingsCell{1})
        eval(['settings.' settingsCell{1}{ii} '= settingsCell{2}{ii};']);
    end
    
end