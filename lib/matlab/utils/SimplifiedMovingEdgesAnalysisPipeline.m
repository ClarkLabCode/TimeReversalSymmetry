function SimplifiedMovingEdgesAnalysisPipeline(symmetries, stim_number, number_for_display, savepath)


stim = strcat('movingEdges_', symmetries);
% Genotype of your flies
genotype = 'Nathan_IsoD1';
%date
%year = 'Combined';
%date = '07_27';

% Get the path to the data directory on your computer
sysConfig = GetSystemConfiguration;
% concatenate those to create the path
dataPath = [sysConfig.dataPath,'/',genotype,'/',stim,...
    ..., '/' date ...
    ];
%dataPath or HollyDataPath
% Your analysis file (you can pass multiple analysis function names as a
% cell)
% "PlotTimeTraces" clips out peri-stimulus turning and walking speed time
% traces, averages over repetitions / flies, and plot group mean time traces
% with standard error around it.
% Other analysis functions can be found under analysis/analysisFiles
analysisFiles={'PlotTimeTraces_Nathan'};

figLeg={'sine'	'Stim1', 'Stim1 t',	'Stim1 c' 'Stim1 ct', 'Stim2', 'Stim2 t', 'Stim2 c', 'Stim2 ct'};

% Prepare input arguments for RunAnalysis function
args = {'analysisFile',analysisFiles,...
        'dataPath',dataPath,...
        'combOpp',1,'figLeg',figLeg}; % combine left/right symmetric parameters? (defaulted to 1)
%'figLeg',figLeg

% Run the thing
iso_out = RunAnalysis(args{:}, 'ttSnipShift', -1000);

close all
%,...'figLeg',figLeg
%%
timeXiso = iso_out.analysis{1}.timeX/1000; % converting ms to s

meanmat_iso = iso_out.analysis{1}.respMatPlot;%mean turning response and walking response over time - one dimension is the stimulus
semmat_iso = iso_out.analysis{1}.respMatSemPlot; %standard error - on

beh=1;


%Integrate over time using individual fly data
nFly = length(iso_out.analysis{1}.indFly) %the number of flies
indmat = [];
for ff = 1:nFly
    % needs some reformatting from cell to matrix...
    thisFlyMat = cell2mat(permute(iso_out.analysis{1}.indFly{ff}.p8_averagedRois.snipMat,[3,1,2]));
    indmat(:,:,ff) = thisFlyMat(:,:,1); % only care about turning here...
end

if stim_number == 1
    meanmat_iso = meanmat_iso(:, 2:5, :);
    semmat_iso = semmat_iso(:, 2:5, :);
    indmat = indmat(:, 2:5, :);
elseif stim_number == 2
    meanmat_iso = meanmat_iso(:, 6:9, :);
    semmat_iso = semmat_iso(:, 6:9, :);
    indmat = indmat(:, 6:9, :);
end

figLeg={['$S_{' num2str(number_for_display) '}$ '], ['$_\Theta S_{' num2str(number_for_display) '}$ '], ['$_\Gamma S_{' num2str(number_for_display) '}$ '], ['$_{\Gamma\Theta} S_{' num2str(number_for_display) '}$ ']};

%%
TFlipFigLeg = {['$S_{' num2str(number_for_display) '}$'], ['$_\Gamma S_{' num2str(number_for_display) '}$']};
CFlipFigLeg = {['$S_{' num2str(number_for_display) '}$'], ['$_\Theta S_{' num2str(number_for_display) '}$']};
CTFlipFigLeg = {['$S_{' num2str(number_for_display) '}$'], ['$_\Gamma S_{' num2str(number_for_display) '}$']};

TUnflippedIndmat = indmat(:,[1 3],:);
TFlippedIndmat = indmat(:,[2 4],:);

CUnflippedIndmat = indmat(:,[1 2],:);
CFlippedIndmat = indmat(:,[3 4],:);

CTUnflippedIndmat = indmat(:, [1 3], :);
CTFlippedIndmat = indmat(:, [4 2], :);

TCombinedIndmat = TUnflippedIndmat + TFlippedIndmat;
CCombinedIndmat = CUnflippedIndmat - CFlippedIndmat;
CTCombinedIndmat = CTUnflippedIndmat + CTFlippedIndmat;

allFliesTCombined = sum(TCombinedIndmat, 3)./nFly;
allFliesCCombined = sum(CCombinedIndmat, 3)./nFly;
allFliesCTCombined = sum(CTCombinedIndmat, 3)./nFly;


stderrorsTCombined = std(TCombinedIndmat, 0, 3) ./sqrt(size(TCombinedIndmat, 3));
stderrorsCCombined = std(CCombinedIndmat, 0, 3) ./sqrt(size(CCombinedIndmat, 3));
stderrorsCTCombined = std(CTCombinedIndmat, 0, 3) ./sqrt(size(CTCombinedIndmat, 3));


%%
    startTime = 0.05;
    endTime = 3.05;

    indeces = find(timeXiso >= startTime & timeXiso < endTime);
    numValues = length(indeces);

    indFlyTCombinedMeanTurning = permute(sum(TCombinedIndmat(indeces, :, :))./numValues, [3, 2, 1]);

    TCombinedMeanTurning = sum(indFlyTCombinedMeanTurning)./nFly;
    TCombinedMeanTurningStderrors = std( indFlyTCombinedMeanTurning ) ./ sqrt( length( indFlyTCombinedMeanTurning )); 


    indFlyCCombinedMeanTurning = permute(sum(CCombinedIndmat(indeces, :, :))./numValues, [3, 2, 1]);

    CCombinedMeanTurning = sum(indFlyCCombinedMeanTurning)./nFly;
    CCombinedMeanTurningStderrors = std( indFlyCCombinedMeanTurning ) ./ sqrt( length( indFlyCCombinedMeanTurning ));


    indFlyCTCombinedMeanTurning = permute(sum(CTCombinedIndmat(indeces, :, :))./numValues, [3, 2, 1]);

    CTCombinedMeanTurning = sum(indFlyCTCombinedMeanTurning)./nFly;
    CTCombinedMeanTurningStderrors = std( indFlyCTCombinedMeanTurning ) ./ sqrt( length( indFlyCTCombinedMeanTurning ));
%%
color = linspecer(4);
set(groot, 'DefaultAxesTickLabelInterpreter','tex');
set(groot,'defaultAxesXTickLabelRotationMode','auto');
%set(groot, 'DefaultTextInterpreter', 'tex')

fig = figure;
%raw turning traces
fig_position = 1
for i = [1 3 2 4]
    ax{fig_position} = subplot(12, 2, [6*fig_position-5 6*fig_position - 3]);
    hold on
    %title([figLeg{i}], 'Interpreter', 'latex')
    
    if i == 1
        ylabel(['$R[S]$ $(^{\circ}/s)$'], 'Interpreter', 'latex')
    else
        set(ax{fig_position}, 'YTickLabel', [])
    end

    PlotXvsY(timeXiso, meanmat_iso(:, i, beh), 'error', semmat_iso(:, i, beh), 'plotColor', color(i, :))

    xlim([-1 4])

    yline(0, '--', 'Color', 'black', 'LineWidth', 1)
    %xline(0, '--', 'Color', [0.5 0.5 0.5]) % vertical 0 line
    %xline(3, '--', 'Color', [0.5 0.5 0.5]) %vertical 3 line


    hold off
    fig_position = fig_position + 1;
end
%xlabel(ax{1}, 'time')

%
%ylabel(ax{4}, ['turning (' char(186) '/s)']);



%t flips
%t bar plot 
ax{6} = subplot(12, 2, [10 14]);
title({"Time reversal","symmetry breaking", " ", " ", " "}, 'Interpreter', 'tex', 'FontWeight','Normal')
ylabel({'$R[S] + R[_\Theta S]$ $(^{\circ}/s)$'}, 'Interpreter', 'latex', 'FontWeight','Normal')

jitter = rand(size(indFlyTCombinedMeanTurning(:, 1)))./8 - 0.06125;

%b_t = bar([TCombinedMeanTurning([1 2])], 'FaceColor', 'flat');

%for i = [1, 2]
%    b_t.CData(i, :) = color(2, :);
%end

set(gca, 'XTickLabel', {TFlipFigLeg{1}, TFlipFigLeg{2}}, 'xtick', [1 2])
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis

hold on
%ylabel(['mean turning (' char(186) '/s)']);
ConfAxis;

plot(jitter + 1 + 0.2, indFlyTCombinedMeanTurning(:, 1), 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);
plot(jitter + 2 - 0.2, indFlyTCombinedMeanTurning(:, 2), 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);

er_t = errorbar([1 2], [TCombinedMeanTurning([1 2])], [TCombinedMeanTurningStderrors([1 2])], "o", 'Color', color(2, :), 'MarkerFaceColor', color(2, :));

er_t.LineWidth = 3;
er_t.MarkerSize = 5;
er_t.CapSize = 0;

yline(0, '--', 'Color', 'black', 'LineWidth', 1);


c = num2cell(indFlyTCombinedMeanTurning(:, [1 2]), 1);
[h, significance_uncorrected] = cellfun(@ttest, c);
%significance = significance_uncorrected .*6;
%significance_uncorrected
all_significance_uncorrected([1, 2]) = significance_uncorrected;
limits = ylim;


%c flips
%c bar plot 
ax{5} = subplot(12, 2, [2 6]);
title({"Contrast reversal","symmetry breaking", " ", " ", " "}, 'Interpreter', 'tex', 'FontWeight','Normal')
ylabel('$R[S] - R[_\Gamma S]$ $(^{\circ}/s)$', 'Interpreter', 'latex', 'FontWeight','Normal')

jitter = rand(size(indFlyTCombinedMeanTurning(:, 1)))./10 - 0.05;

%b_c = bar([CCombinedMeanTurning([1 2])], 'FaceColor', 'flat');

%for i = [1, 2]
%    b_c.CData(i, :) = color(3, :);
%end

set(gca, 'XTickLabel', {CFlipFigLeg{1}, CFlipFigLeg{2}}, 'xtick', [1 2])
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis

hold on
ConfAxis;

plot(jitter + 1 + 0.2, indFlyCCombinedMeanTurning(:, 1), 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);
plot(jitter + 2 - 0.2, indFlyCCombinedMeanTurning(:, 2), 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);

er_c = errorbar([1 2], [CCombinedMeanTurning([1 2])], [CCombinedMeanTurningStderrors([1 2])], "o", 'Color', color(3, :), 'MarkerFaceColor', color(3, :));

er_c.LineWidth = 3;
er_c.MarkerSize = 5;
er_c.CapSize = 0;

yline(0, '--', 'Color', 'black', 'LineWidth', 1);

c = num2cell(indFlyCCombinedMeanTurning(:, [1 2]), 1);
[h, significance_uncorrected] = cellfun(@ttest, c);
%significance = significance_uncorrected .*6;
%significance_uncorrected
all_significance_uncorrected([3, 4]) = significance_uncorrected;


%ct flips
%ct bar plot 
ax{7} = subplot(12, 2, [18 22]);
title({"Time-contrast reversal","symmetry breaking", " ", " ", " "}, 'Interpreter', 'tex', 'FontWeight','Normal')
ylabel('$R[S] + R[_{\Gamma\Theta} S]$ $(^{\circ}/s)$', 'Interpreter', 'latex', 'FontWeight','Normal')
jitter = rand(size(indFlyTCombinedMeanTurning(:, 1)))./10 - 0.05;

% b_ct = bar([CTCombinedMeanTurning([1 2])], 'FaceColor', 'flat');
% 
% for i = [1, 2]
%     b_ct.CData(i, :) = color(4, :);
% end

set(gca, 'XTickLabel', {CTFlipFigLeg{1}, CTFlipFigLeg{2}}, 'xtick', [1 2])
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis

hold on
%ylabel(['mean turning (' char(186) '/s)']);
ConfAxis;

plot(jitter + 1 + 0.2, indFlyCTCombinedMeanTurning(:, 1), 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);
plot(jitter + 2 - 0.2, indFlyCTCombinedMeanTurning(:, 2), 'ko', 'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', [0.7 0.7 0.7]);

er_ct = errorbar([1 2], [CTCombinedMeanTurning([1 2])], [CTCombinedMeanTurningStderrors([1 2])], "o", 'Color', color(4, :), 'MarkerFaceColor', color(4, :));

er_ct.LineWidth = 3;
er_ct.MarkerSize = 5;
er_ct.CapSize = 0;

yline(0, '--', 'Color', 'black', 'LineWidth', 1);

c = num2cell(indFlyCTCombinedMeanTurning(:, [1 2]), 1);
[h, significance_uncorrected] = cellfun(@ttest, c);
%significance = significance_uncorrected .*6;
%significance_uncorrected
all_significance_uncorrected([5, 6]) = significance_uncorrected;


for i = 1:7
    ylim(ax{i}, 'padded');
end    

linkaxes([ax{1} ax{2} ax{3} ax{4}], 'y');

%linkaxes([ax{5} ax{6} ax{7}], 'y');


for i = [1]
    max_lim = max(abs(ylim(ax{i})));
    ylim(ax{i}, [-1*max_lim, max_lim]);
end

for i = [5 6 7]
    xlim(ax{i}, [0.7, 2.3]);
    ax{i}.XAxis.FontSize= 14;
    ax{i}.XLabel.FontSize = 14;
    ax{i}.Title.FontSize = 12;
    ax{i}.YAxis.FontSize= 12;
    ax{i}.YLabel.FontSize = 14;


    %limits = ylim(ax{i});
    %ax{i}.XLabel.Position = [1.5  limits(2) + (limits(2) - limits(1)) * 0.1];
    %ax{i}.XLabel.VerticalAlignment = 'bottom';

end

for i = [1 2 3 4]
    hold(ax{i}, 'on')
    max_lim = max(abs(ylim(ax{i})));
    patch(ax{i}, [0 0 3 3], [-1*max_lim, 1*max_lim, 1*max_lim, -1*max_lim], [0.3 0.3 0.3], 'FaceAlpha',0.2, 'LineStyle', 'none')
    plot(ax{i}, [0 1], [-0.75 * max_lim, -0.75 * max_lim], 'black', 'LineWidth', 4);
    text(ax{i}, 0.5, -0.5 * max_lim, '1s', 'FontSize', 12, 'HorizontalAlignment', 'center');
    ax{i}.Title.FontSize = 15;

    ax{i}.YAxis.FontSize= 12;
    ax{i}.YLabel.FontSize = 14;


    ylim(ax{i}, [-1*max_lim, max_lim])
    hold(ax{i}, 'off')

    set(ax{i},'XTickLabel',[]);
    ax{i}.XAxis.Visible = 'off';


end

%b_list = {b_t, b_c, b_ct};
ax_list = {ax{5}, ax{6}, ax{7}};

all_significance_uncorrected
all_significance = bonf_holm(all_significance_uncorrected, 0.05)
for num = [1 2 3]
    ax_nums = [2 1 3];
    ax_num = ax_nums(num);
    %b = b_list{num};
    ax_i = ax_list{ax_num};
    
    significance = all_significance([2*num - 1, 2*num]);

    %hold(ax{i}, 'on')
    set(gcf, 'CurrentAxes', ax_i);

    for j = 1:length(significance)
    %y = b.YData(j) + (2*er.YPositiveDelta(j)*((b_ct.YData(j)>=0) - 0.5) + (ax.YLim(2) - ax.YLim(1))/8 * ((b_ct.YData(j)>=0) - 0.7));
        sigstar([j j], significance(j));
    end

end
    

fontsize(scale = 1)

for i = 1:7
    view(ax{i}, [90 90]);
    set(ax{i}, 'YAxisLocation','right');

end   

for i = 1:4
            position_data = get(ax{i}, 'Position');
            %position_data(2) = position_data(2) + position_data(4) * 0.1 * (7-i);
            position_data(3) = position_data(3) * 0.7;
            position_data(2) = position_data(2) - position_data(4) * 0.07 * (4-i);
            position_data(4) = position_data(4) * 1.21;
            set(ax{i}, 'Position', position_data);
end

for i = 5:7
            position_data = get(ax{i}, 'Position');
            position_data(2) = position_data(2) + position_data(4) * 0.02 * (7-i);
            position_data(4) = position_data(4) * 0.7;
            position_data(1) = position_data(1) + position_data(3) * 0.15;
            position_data(3) = position_data(3) * 0.85;
            set(ax{i}, 'Position', position_data);
end

set(gcf, 'WindowStyle', 'normal')
set(gcf,'Position',[80 0 350 800])


saveas(gcf, strcat(savepath, 'S', num2str(number_for_display), '_', symmetries, '_', num2str(stim_number)));
saveas(gcf, strcat(savepath, 'S', num2str(number_for_display), '_', symmetries, '_', num2str(stim_number), '.pdf'));
saveas(gcf, strcat(savepath, 'S', num2str(number_for_display), '_', symmetries, '_', num2str(stim_number), '.png'));
%close all;