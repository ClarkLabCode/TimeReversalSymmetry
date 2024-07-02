function [flyResp,epochs,params,stim,flyIds,rigIds,numTotalFlies, rawResp,rawStim,paramPaths,finalPaths,runDetails] =  BuildDANDIDataset(stimfileName, datapath, savepath, varargin)
%https://neurodatawithoutborders.github.io/matnwb/tutorials/html/intro.html#H_FF8B1A2D
%https://neurodatawithoutborders.github.io/matnwb/doc/index.html
%https://github.com/NeurodataWithoutBorders/matnwb

sampleRate = 60; %frames per second


%Need to pass in datapath.
[rawResp,rawStim,paramPaths,finalPaths,runDetails] = LoadData({datapath} ,[], [], []);
%we only care about finalPaths I believe?

 [flyResp,epochs,params,stim,flyIds,rigIds,numTotalFlies] = ReadBehavioralData({datapath});


 %What does rawResp actually hold?? What is 19?
 %Use flyIds to match up flies to rigs?/samples run?

 %match to runDetails?

 %extract date and time
 for i = 1:length(runDetails)
    singleFlyIds = GetFlyIds({runDetails{i}});
    flyIdsPerRun = RepFlyIds(singleFlyIds,5);
    
    flyIdIndeces = [];
    for id = flyIdsPerRun
        flyIdIndeces = horzcat(flyIdIndeces, find(flyIds == id));
    end

    %get date and time
    fileParts = split(finalPaths.param{i}, '\');
    y_cell= fileParts(length(fileParts)-3);
    md_cell = split(fileParts(length(fileParts)-2), '_');
    hms_cell = split(fileParts(length(fileParts)-1), '_');
    year = str2num(y_cell{1});
    month = str2num(md_cell{1});
    day = str2num(md_cell{2});
    hour = str2num(hms_cell{1});
    min = str2num(hms_cell{2});
    sec = str2num(hms_cell{3});
    

    epochNames = arrayfun(@(s) s.epochName, params{1}, 'UniformOutput', false);


    
    % Read all optional input arguments
    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end


    
    for flyIndex = flyIdIndeces
        flyIndex;
        flyId = flyIds(flyIndex);
        resp = flyResp{flyIndex};
        epoch = uint8(epochs{flyIndex} - 1);



        nwb = NwbFile( ...
            'file_create_date', datetime('now', 'TimeZone', 'local'),...
            'general_experiment_description', 'optomotor turning of Drosophila on air-supported balls when presented with visual stimuli', ...
            'general_keywords', {'motion detection', 'symmetry', 'symmetry breaking', 'algorithm', 'Drosophila'}, ...
            'session_description', [stimfileName, ' ' num2str(month) '-' num2str(day) '-' num2str(year) ' ' num2str(hour) ':' num2str(min) ':' num2str(sec)],...
            'identifier', [stimfileName, ' ' num2str(month) '-' num2str(day) '-' num2str(year) ' ' num2str(hour) ':' num2str(min) ':' num2str(sec) ' ' num2str(flyId)], ....
            'session_start_time', datetime(year, month, day, hour, min, sec, 'TimeZone', 'America/New_York'), ...
            'timestamps_reference_time', datetime(year, month, day, hour, min, sec, 'TimeZone', 'America/New_York'), ...
            'general_experimenter', 'Wu, Nathan', ... % optional
            'general_institution', 'Yale University', ... % optional
            'general_related_publications', {'https://doi.org/10.1101/2024.06.08.598068'}, ... % optional
            'general_lab', 'Clark Lab'...
            );

        nwb.general_subject = types.core.Subject(...
            'age', 'P2D', ...
            'description', [stimfileName ' fly'], ...
            'sex', 'F', ...
            'species', 'Drosophila melanogaster', ...
            'strain', 'Oregon R', ...
            'subject_id', [stimfileName '_' num2str(flyId)] ...
            );

        response_series = types.core.TimeSeries( ...
            'description', [stimfileName, ' fly no ' num2str(flyId) ' optomotor turning on an air-supported ball'],...
            'data', resp(:, :, 1), ...
            'data_unit', '°/s', ...
            'starting_time', 0, ...
            'starting_time_rate', 60, ...
            'control', epoch, ...
            'control_description', epochNames ...
            );

        nwb.acquisition.set('TimeSeries', response_series);

        %OpticalSeries can be used to store stimulus data...
        %https://neurodatawithoutborders.github.io/matnwb/tutorials/html/images.html

        nwbExport(nwb, [savepath stimfileName '_' num2str(flyId) '.nwb']);
    end

 end

%epoch names are in params
%Method:
%Loop through flyIds or flyResp?

%epochs, resp, flyIds change between different flies. 
%Also the session start time


