
clear all; 
close all;


%------------------------------------------------------------------------
% Plot histogramms

% Train, validation: 
jrca1 = load('jrc_train_valid.mat', 'jrca');
z2a1  = load('jrc_train_valid.mat', 'z2a');
jrca1 = jrca1.jrca;
z2a1  = z2a1.z2a;

% Test:  
jrca2 = load('jrc_test.mat', 'jrca');
z2a2  = load('jrc_test.mat', 'z2a');
jrca2 = jrca2.jrca;
z2a2  = z2a2.z2a;

% Test no depth sample:  
jrca3 = load('jrc_test_nodepth.mat', 'jrca');
z2a3  = load('jrc_test_nodepth.mat', 'z2a');
jrca3 = jrca3.jrca;
z2a3  = z2a3.z2a;


% JRC
figure(1); clf; 
subplot(1, 2, 2);
histogram(jrca1, 100); hold on; 
histogram(jrca2, 100); 
histogram(jrca3, 100); hold off;
xlabel('JRC'); ylabel(['Frequency']); 
legend('Train and Cross-validation data', 'Test data');
legend boxoff;

% z2
subplot(1, 2, 1);
histogram(z2a1, 100); hold on;
histogram(z2a2, 100); 
histogram(z2a3, 100); hold off;
xlabel('Z_2'); ylabel(['Frequency']); 
xlim([0, 1.5]);
legend('Train and Cross-validation data', 'Test data 611No2', 'Test data no depth');
legend boxoff;

ss = get(groot, 'Screensize'); 
set(1, 'Position', [ss(1)*1900, ss(2)-50, 900, 730]);

