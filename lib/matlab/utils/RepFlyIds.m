function repeatedFlyIds = RepFlyIds(flyIds,numRep)
    % this will duplicate fly ids for each fly on the rig, and set the
    % least significant bits to the rig number of the fly

    repeatedFlyIds = flyIds(1,ceil(1/(numRep*2):1/numRep:end));
    
    for ii = 1:numRep
        for jj = ii:numRep:length(repeatedFlyIds)
            bitSet = fliplr(dec2bin(ii-1,length(dec2bin(numRep))));
            
            for bb = 1:length(bitSet)
                repeatedFlyIds(jj) = bitset(repeatedFlyIds(jj),bb,str2double(bitSet(bb)),'uint64');
            end
        end
    end
end