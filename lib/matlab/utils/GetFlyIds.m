function flyIds = GetFlyIds(runDetails)
    if isstruct(runDetails{1})
        flyIds = uint64(zeros(1,length(runDetails)));

        for rr = 1:length(runDetails)
            if ischar(runDetails{rr}.flyId)
                flyIds(rr) = str2num(['uint64(' runDetails{rr}.flyId ')']);
            else
                flyIds(rr) = runDetails{rr}.flyId;
            end
        end
    elseif ischar(runDetails{1})
        flyIds = GetFlyIdFromDatabase(runDetails);
    end
end