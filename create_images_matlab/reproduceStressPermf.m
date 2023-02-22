% Script to reproduce stress-permf for 347 ML stress-permf project
% Test 37 - 5 data is here as well

clear all 
close all


%--------------------------------------------------------------------------------------------
% Specify manually: 


% Load stress/permf
filepath = 'D:\expfracML\data\TrainTest247_processed\Test37\stress.csv'; 
data = readtable(filepath, 'HeaderLines', 1); 
stress_all = table2array(data);  
clear data
filepath = 'D:\expfracML\data\TrainTest247_processed\Test37\permf.csv'; 
data = readtable(filepath, 'HeaderLines', 1); 
permf_all = table2array(data);  
clear data

fprintf('Data is assumed cleaned from the empty subcases \n');


%--------------------------------------------------------------------------------------------
% Find the maxInitStress/minFinalStress in all cases and subcases

maxInitStress = max(stress_all(:, 1)); 

for j=1:length(stress_all(:, 1))

    stress1 = stress_all(j, :);
    stress1(isnan(stress1)) = []; 
    final_stress1_all(j, 1) = stress1(end);
    
    
end
minFinalStress = min(final_stress1_all(:)); 
clear stress1 permf1 final_stress1_all

fprintf('maxInitStress in 247 = 7348.98040993290 \n'); 
fprintf('minFinalStress in 247 = 50000053.5808048 \n'); 


omitcases = [15,16,17,19,33]; % In testing, as maxInitStress larger than in 247


%--------------------------------------------------------------------------------------------


rapers = [];
for i = 1:length(stress_all(:, 1))
    
    if ismember(i, omitcases)
         fprintf(['Omitting subcase number= ', num2str(i), '\n']);
        continue
    end
    
    
    fprintf(['subcase number= ', num2str(i), '\n']);
    
    mstresshistbc = stress_all(i, :);
    permf = permf_all(i, :); 
    
    % Delete NaNs for single curves
    mstresshistbc(isnan(mstresshistbc)) = []; 
    permf(isnan(permf)) = []; 
    
    assert(length(permf) == length(mstresshistbc), 'Lengths are not equal'); 
    
    
    %--------------------------------------------------------------------------------------------
    % Reproduce curves of stress-perm for the 'mean and std calculation' below
    
    curve = [mstresshistbc', permf']; 
    rcurve = reproduceCurveVert(curve, 'xmin', maxInitStress , 'xmax', minFinalStress); 
    
    rstress_all(i, :) =  rcurve(:, 1)'; 
    rpermf_all(i, :) = rcurve(:, 2)';
    
    clear rcurve curve 

end 


fprintf('Finished \n');

figure(3)
j = 37;  
plot(stress_all(j, :), permf_all(j, :)); hold on;
plot(rstress_all(j, :), rpermf_all(j, :), 'x'); hold off;
xlabel('Stress, Pa');
ylabel('Kf, mD');
title('Example curve 37 original and reproduced'); 



%--------------------------------------------------------------------------------------------
% APPENDIX

% Plot all stress-permf curves
% figure(2)
% for i = 1:length(stress_all)
%     fprintf('i: %d\n', i)
%     plot(stress_all{i, :}, permf_all{i, :}); hold on; 
%     fprintf('..... \n');
% end
% hold off; 
% xlabel('Stress, Pa');
% ylabel('Kf, mD');


