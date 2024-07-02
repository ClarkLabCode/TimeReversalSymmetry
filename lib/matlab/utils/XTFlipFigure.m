function XTFlipFigure(figLeg, pairs, mode, stimulus_duration, save_filepath, stim, genotype, year, xtPlot_year, varargin)
    close all


%% Running "RunAnalysis"

    % RunAnalysis is the outermost wrapper function for data analysis both for
    % behavior and imaging datout. It expects "analysisFile" and "dataPath"
    % arguments at least. 

    % Name of your stimulus

    % Genotype of your flies
    %date
    %date = '08_02'; 
    % Get the path to the data directory on your computer
    sysConfig = GetSystemConfiguration;
    % concatenate those to create the path
    %dataPath = [sysConfig.dataPath,'/',genotype,'/',stim, '/', year, '/' date];
    dataPath = [sysConfig.dataPath,'/',genotype,'/',stim, '/', year];

    % Your analysis file (you can pass multiple analysis function names as a
    % cell)
    % "PlotTimeTraces" clips out peri-stimulus turning and walking speed time
    % traces, averages over repetitions / flies, and plot group mean time traces
    % with standard error around it.
    % Other analysis functions can be found under analysis/analysisFiles
    analysisFiles={'PlotTimeTraces_Nathan'};

    % figLeg={'Cross0CenterR','	Cross0CenterL','	Pro15CenterL','	Pro15CenterR',	'Pro45CenterL','	Pro45CenterR','	Pro75CenterL',...
    %     'Pro75CenterR','	Pro105CenterL','	Pro105CenterR','	Pro120CenterL','	Pro120CenterR',...
    %     'Reg15CenterL','	Reg15CenterR','	Reg45CenterL','	Reg45CenterR','	Reg75CenterL','	Reg75CenterR',...
    %     'Reg105CenterL','	Reg105CenterR','	Reg120CenterL','	Reg120CenterR'}

    % Prepare input arguments for RunAnalysis function
    args = {'analysisFile',analysisFiles,...
            'dataPath',dataPath,...
            'combOpp',1,'figLeg',figLeg}; % combine left/right symmetric parameters? (defaulted to 1)
    %'figLeg',figLeg

    % Run the thing
    iso_out = RunAnalysis(args{:}, varargin{:}, 'ttSnipShift', -1000);
    %,...'figLeg',figLeg
    close all


    %%

    color = linspecer(size(figLeg, 2));

    timeXiso = iso_out.analysis{1}.timeX/1000; % converting ms to s
    meanmat_iso = iso_out.analysis{1}.respMatPlot;%mean turning response and walking response over time - one dimension is the stimulus
    semmat_iso = iso_out.analysis{1}.respMatSemPlot; %standard error - on

    %%
    %Integrate over time using individual fly data
    nFly = length(iso_out.analysis{1}.indFly) %the number of flies
    indmat = [];
    for ff = 1:nFly
        % needs some reformatting from cell to matrix...
        thisFlyMat = cell2mat(permute(iso_out.analysis{1}.indFly{ff}.p8_averagedRois.snipMat,[3,1,2]));
        indmat(:,:,ff) = thisFlyMat(:,:,1); % only care about turning here...
    end

    startTime = 0.05;
    endTime = stimulus_duration + 0.05;

    indeces = find(timeXiso >= startTime & timeXiso < endTime);
    numValues = length(indeces);
    indFlyMeanTurning = permute(sum(indmat(indeces, :, :))./numValues, [3, 2, 1]);


    meanTurning = sum(indFlyMeanTurning)./nFly;
    stderrors = std( indFlyMeanTurning ) ./ sqrt( length( indFlyMeanTurning )); 

    %%
    %Subtract/add pairs of t flipped stimuli.
    %CHANGE THESE

    tFlip_indmat = NaN(size(indmat, 1), length(pairs), nFly);

    for i = 1:length(pairs)
        pair = pairs{i};
        first_index = pair(1);
        second_index=pair(2);
        
        %if first_index and second_index are the same, need to select half
        %of the flies to be the original and the other half to be the flip.

        if strcmp(mode, 'xt_flip')
            tFlip_indmat(:,i,:) = indmat(:,first_index,:) - indmat(:,second_index,:);
        elseif strcmp(mode, 't_flip')
            tFlip_indmat(:,i,:) = indmat(:,first_index,:) + indmat(:,second_index,:);
        else
            disp('ERROR - mode should be xt_flip or t_flip');
        end

    end
    %%
    %tFlip_indmat_allFlies = sum(tFlip_indmat, 3)./nFly;
    tFlip_indmat_allFlies = mean(tFlip_indmat, 3);
    tFlip_indmat_allFlies_stderrors = std(tFlip_indmat, 0, 3) ./sqrt(size(tFlip_indmat, 3));

    %%
    t_flip_indFlyMeanTurning = permute(sum(tFlip_indmat(indeces, :, :))./numValues, [3, 2, 1]);


    t_flip_meanTurning = sum(t_flip_indFlyMeanTurning)./nFly;
    t_flip_stderrors = std( t_flip_indFlyMeanTurning ) ./ sqrt( length( t_flip_indFlyMeanTurning ));


    %%      
    %MAY NEED TO CHANGE THIS
    xtPlot_dataPath = fullfile('xtplot', stim, xtPlot_year);
    [epochXtPlot, frameRate] = AnalyzeXtPlot(xtPlot_dataPath, 0, 0);
    frameRate;
    tRes = 1/(frameRate);

    indexToEpoch = @(x) 4.*x - 2;



    %code in AnalyzeXtPlot to display:

    % for ee = 1:numEpochs
    %             pixelSize = 360/(size(XtPlot,2));
    %             tickX = (0:pixelSize:360-pixelSize)';
    %             tickY = (0:tRes:tRes*(size(XtPlot,1)-1))'*1000;
    %             
    %             figH=MakeFigure;
    %             figTitle = ['xtPlot epoch ' num2str(ee)];
    %             set(figH,'name',figTitle,'NumberTitle','off');
    %             title(figTitle);
    %             imagesc(tickX,tickY,XtPlot);
    %             ConfAxis('tickX',tickX,'tickLabelX',tickX,'tickY',tickY,'tickLabelY',round(tickY*10)/10,'labelX',['space (' char(186) ')'],'labelY','time (ms)');
    %             colormap(gray);
    %         end

    %%
    %THINGS THAT ARE ADJUSTABLE
    x_limits = [-1, stimulus_duration + 1];


    color = linspecer(2);
    for j = 1:size(pairs, 2) %skip the ones that are the same for now
        fig = figure;
        pair = pairs{j};
        first_index = pair(1);
        second_index=pair(2);
        
        set(fig, 'defaultAxesColorOrder', color);

        %ax{1} = subplot(8,3,[16, 23]);
        ax{1} = subplot(5, 7, [12, 21]);

        % Use in-house prettier plot functions for visualization...
        hold on
        %title ('Individual time traces')
        % showing only first three epochs 
        %PlotXvsY(timeXiso,meanmat_iso(:,j,1),'error',semmat_iso(:,j,1),'plotColor',color(1,:));
        %legend('Genetic Control','T4/T5 Silenced','Empty Shts')
        
        beh = 1;
        if strcmp(mode, 't_flip')
            PlotXvsY(timeXiso,meanmat_iso(:,[first_index second_index], beh),'error', semmat_iso(:,[first_index, second_index], beh),'plotColor',color(:,:));
        elseif strcmp(mode, 'xt_flip')
            PlotXvsY(timeXiso,[meanmat_iso(:,first_index, beh), meanmat_iso(:, second_index, beh).*-1],'error', semmat_iso(:,[first_index, second_index], beh),'plotColor',color(:,:));
        end
        
        xlim(x_limits)
        yline(0, '--', 'color', 'black', 'LineWidth', 1)
        % xlim([-0.5 2])
        % ylim([-40 40])
        %ylim([-40 110])% Change here! So that you are plotting the correct time window

        
        %xlabel('time (s)')
        ylabel(['turning (' char(186) '/s)']);
        set(ax{1},'XTickLabel',[]);
        set(ax{1},'xtick',[]);
        ax{1}.XAxis.Visible = 'off';

        %legend("Time forwards", 'Time reversed')


        hold off


        %%
        ax{2} = subplot(5, 7, [26 28]);
        
        %categories = categorical(figLeg([first_index, second_index]));

        % if strcmp(mode, 'xt_flip')
        %     b = bar(meanTurning([first_index, second_index]), 'FaceColor', 'flat');
        % elseif strcmp(mode, 't_flip')
        %     b = bar([meanTurning(first_index), meanTurning(second_index).*-1], 'FaceColor', 'flat');
        % end
        % 
        % for i = [1, 2]
        %     b.CData(i, :) = color(i, :);
        % end
        ylim([0.2 2.8])

        set(ax{2}, 'YTickLabel', {"Time forward", '', "Difference"}, 'ytick', [0.6 2.4 3], 'XColor', color(1, :), 'YColor', color(1, :),'xTickLabelRotation', 0)%, %'XTickLabelRotation', 90, 'xaxisLocation', 'top')
        
        %set(ax{2}, 'xtick', [])

        hold on

        %data gets reset
        data = indFlyMeanTurning(:,[first_index, second_index]).';

        jitter = rand(size(data))./2.5 - 0.2;
        jitter(2, :) = jitter(2, :) .* -1;

       
        if strcmp(mode, 'xt_flip')
            data = indFlyMeanTurning(:,[first_index, second_index]).';

            subtracted_data = data(2, :) - data(1, :);

            %s = plot(jitter + repmat([1; 2], [1, nFly]), data, 'ko-', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);
            s = plot(data, jitter + repmat([1; 2], [1, nFly]), 'ko-', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);

            %s_sub = plot(rand([1, nFly])./2.5 - 0.2 + 3, subtracted_data, 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);

            er1 = errorbar(meanTurning(first_index), 0.6, stderrors(first_index), "o", 'horizontal', 'Color', color(1, :), 'MarkerFaceColor', color(1, :));%'LineStyle', 'none');
            er2 = errorbar(meanTurning(second_index), 2.4, stderrors(second_index), "o", 'horizontal', 'Color', color(2, :), 'MarkerFaceColor', color(2, :));%'LineStyle', 'none');
            %er_sub = errorbar(3, mean(subtracted_data), std(subtracted_data)./sqrt(nFly), "o", 'Color', mean(color, 1), 'MarkerFaceColor', mean(color, 1));%'LineStyle', 'none');

            %s = plot([1, 2], indFlyMeanTurning(:,[first_index, second_index]).', 'ko-', 'Color', [0.5 0.5 0.5]);
            %er = errorbar([1, 2],meanTurning([first_index, second_index]),stderrors([first_index, second_index])); 
        elseif strcmp(mode, 't_flip')
            data = [indFlyMeanTurning(:,first_index), indFlyMeanTurning(:, second_index) * -1].';

            subtracted_data = data(2, :) - data(1, :);

            s = plot(data, jitter + repmat([1; 2], [1, nFly]), 'ko-', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);
            %s_sub = plot(rand([1, nFly])./2.5 - 0.2 + 3, subtracted_data, 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);

            er1 = errorbar(meanTurning(first_index), 0.6, stderrors(first_index), "o", 'horizontal', 'Color', color(1, :), 'MarkerFaceColor', color(1, :));%'LineStyle', 'none');
            er2 = errorbar(meanTurning(second_index).*-1, 2.4, stderrors(second_index), "o", 'horizontal', 'Color', color(2, :), 'MarkerFaceColor', color(2, :));%'LineStyle', 'none');
            %er_sub = errorbar(3, mean(subtracted_data), std(subtracted_data)./sqrt(nFly), "o", 'Color', mean(color, 1), 'MarkerFaceColor', mean(color, 1));%'LineStyle', 'none');

            %s = plot([1, 2], [indFlyMeanTurning(:,first_index), indFlyMeanTurning(:, second_index) * -1].', 'ko-', 'Color', [0.5 0.5 0.5]);
            %er = errorbar([1, 2],[meanTurning(first_index), meanTurning(second_index).*-1],stderrors([first_index, second_index]));
        end

        xline(0, '--', 'color', 'black', 'LineWidth', 1)

        %er.Color = [0 0 0];                            
        %er.LineStyle = 'none';  


        c = num2cell(t_flip_indFlyMeanTurning(:, j), 1);
        [h, significance] = cellfun(@ttest, c);


        %title('Mean turning response')
        %ylabel(['mean turning (' char(186) '/s)']);

        ConfAxis;

        sigstar([0.6, 2.4], significance(1), 1);
        %sigstar([3 3], significance(1));
        %text(0, 0, ['p = ' num2str(significance(1), 3)]);

        er1.LineWidth = 3;
        er2.LineWidth = 3;
        %er_sub.LineWidth = 3;
        er1.MarkerSize = 5;
        er2.MarkerSize = 5;
        %er_sub.MarkerSize = 5;
        er1.CapSize = 0;
        er2.CapSize = 0;
        %er_sub.CapSize = 0;


        %for i = 1:size(s_sub, 1)
        %    s_sub(i).LineWidth = 1;
        %    s_sub(i).MarkerSize = 5;
        %end
        for i = 1:size(s, 1)
            s(i).LineWidth = 1;
            s(i).MarkerSize = 5;
        end        
        
        xl = xlim;
        xt = xticks;

        ax{5} = axes('position',ax{2}.Position, ... % create the second axis, top/right...
            'color', 'none', ...
            'XAxisLocation','bottom', ...
            'XColor', color(2, :), ...
            'YColor', color(2, :), ...
            'ytick', yticks(ax{2}), ...
            'ylim', ylim(ax{2}), ...
            'yTickLabel', ["", "Time reversed"],...
            'xTickLabelRotation', 0 ...
            );

        xlabel(['mean turning (' char(186) '/s)']);

        xlim([max(xl) * -1, min(xl) * -1])
        xt = xt* -1;
        xt = sort(xt);
        xticks(xt);
        set(ax{5}, 'XDir', 'reverse', 'YDir', 'reverse');
        ConfAxis;

        hold off


        for i = [1 2]
            ax{i + 2} = subplot(5, 7, [2*i+6, 2*i + 14]);
            hold on

            % ee = indexToEpoch(pair(i));
            % XtPlot = epochXtPlot{ee};
            % 
            % if (strcmp(mode, 'xt_flip') && i == 2)
            %     XtPlot = flip(XtPlot, 2);
            % end

            ee = indexToEpoch(pair(1));
            XtPlot = epochXtPlot{ee};

            if (i == 2)
                XtPlot = flip(XtPlot, 1);
            end


            pixelSize = 360/(size(XtPlot,2));

            sz = size(epochXtPlot{ee - 1});
            full_XtPlot = vertcat(epochXtPlot{ee - 1}, XtPlot(:, 1:sz(2)), epochXtPlot{ee + 1});
            tickX = (0:pixelSize:360-pixelSize)';
            %tickY = (0:tRes:tRes*(size(XtPlot,1)-1))'*1000;
            tickY = -1 * tRes*(size(epochXtPlot{ee - 1}, 1) - 1):tRes:tRes*(size(XtPlot,1) + size(epochXtPlot{ee + 1}-1, 1));
            % if i == 1
            %     figTitle = ["Time forwards"];
            % else
            %     figTitle = ['Time reversed'];
            % end
            % title(figTitle);
            %imagesc(tickX,tickY,XtPlot);
            imagesc(tickY, tickX, full_XtPlot.');
            ylabel(['space (' char(186) ')']);
            xlim(x_limits)
            ylim([0 90])

            set(ax{i + 2}, 'XTickLabel', []);
            yticks(ax{i + 2}, [0 90]);
            %set(gca, 'XLabel', 'time (ms)', 'YLabel', ['space (' char(186) ')']);

            %ConfAxis('tickX',tickX,'tickLabelX',tickX,'tickY',tickY,'tickLabelY',round(tickY*10)/10,'labelX',['space (' char(186) ')'],'labelY','time (ms)');
            colormap gray
            ConfAxis


            hold off

        end
        linkaxes([ax{1} ax{3} ax{4}], 'x')
        %linkaxes([ax{1} ax{2}], 'y')
        %sgtitle(figLeg{pair(1)})
        title(ax{3}, figLeg{pair(1)}, 'FontWeight','Normal', 'FontSize', 16)
        ax{3}.TitleHorizontalAlignment = 'left';

        set(ax{3}, 'box', 'on', 'XColor', color(1, :), 'YColor', color(1, :), 'LineWidth', 8)
        set(ax{4}, 'box', 'on', 'XColor', color(2, :), 'YColor', color(2, :), 'LineWidth', 8)


        ax{2}.YAxis.FontSize= 14;
        ax{5}.YAxis.FontSize= 14;

        for i = 3:4
            position_data = get(ax{i}, 'Position');
            position_data(3) = position_data(3) * 0.7;
            position_data(1) = position_data(1) + position_data(3) * 0.15;
            set(ax{i}, 'Position', position_data);
        end
        
        for i = [1 3 4]
            view(ax{i},[90 90]) %// instead of normal view, which is view([0 90])
            set(ax{i}, 'YAxisLocation', 'right');
        end

        hold(ax{1}, 'on')
        limits = ylim(ax{1});
        sum_limits = limits(2) - limits(1);
        patch(ax{1}, [0 0 stimulus_duration stimulus_duration], [limits(1), limits(2), limits(2), limits(1)], [0.3 0.3 0.3], 'FaceAlpha',0.2, 'LineStyle', 'none')
        plot(ax{1}, [0 1], [0.125*sum_limits + limits(1), 0.125*sum_limits + limits(1)], 'black', 'LineWidth', 4);
        text(ax{1}, 0.5, 0.22*sum_limits + limits(1), '1s', 'FontSize', 12, 'HorizontalAlignment', 'center');
        ylim(ax{1}, limits);

        
        set(ax{2}, 'XAxisLocation', 'top', 'ydir', 'reverse');

        set(gcf,'Position',[200 200 400 600])
        hold (ax{1}, 'off')

        saveas(gcf, strcat(save_filepath, stim, figLeg{first_index}));
        saveas(gcf, strcat(save_filepath, stim, figLeg{first_index}, '.pdf'));    
        saveas(gcf, strcat(save_filepath, stim, figLeg{first_index}, '.png'));    


    end
end