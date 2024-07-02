function analysisOut = MakeAnalysisReadable(analysisIn)
    for aa = 1:length(analysisIn)
        analysisSubField = ['p' num2str(aa) '_' analysisIn{aa}.name];
        analysisOut.(analysisSubField) = analysisIn{aa};
    end
end