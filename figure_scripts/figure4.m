clear all
all_symmetries = {'T', 'C', 'TC', 'TX', 'CX', 'XTC', 'C,T,CT', 'C,XT,XCT', 'T,XC,XCT', 'XC,XT,CT', 'nosymm_1', 'nosymm_2'};

folder = fileparts(which(mfilename));
savepath = [folder, '/../figures/fig4/moving_edges_processing/'];
addpath([folder, '/..']);
addpath(genpath([folder, '/../lib/matlab']));

%all_symmetries = {'C,XT,XCT', 'T,XC,XCT', 'XC,XT,CT', 'nosymm_1', 'nosymm_2'};

%stim_numbers = {1, 2};

number_for_display = 1;
%number_for_display = 15;


for symmetry_number = 1:length(all_symmetries)
    for stim_number = 1:2
        number_for_display
        %MovingEdgesAnalysisPipeline(all_symmetries{symmetry_number}, stim_number, number_for_display)
        SimplifiedMovingEdgesAnalysisPipeline(all_symmetries{symmetry_number}, stim_number, number_for_display, savepath)

        number_for_display = number_for_display + 1;
    end
end

'done'


