%% 
clear all
close all
% clc

% Does confidence criteria changes after damage?
% Add gain variability
% TODO: run many iterations

rng('shuffle'); % Ensures that different random numbers are generated every time
      
% TIP: Keep all possible 1D variables as row vectors.
% NOTE: Time step should be small relative to the spike rate of a single neuron,
% so that the probability of firing doesn't shoot up. 

% TODO: Add a check to ensure that the firing rate is not too large 
% compared to the time window.

psychometricFns          = zeros(100, 21);
psychometricFnsDamaged   = zeros(100, 21);
psychometricFnsAdjusted  = zeros(100, 21);

psychometricFnsByConf2          = zeros(100, 2, 21); % HC, LC
psychometricFnsDamagedByConf2   = zeros(100, 2, 21); % HC, LC
psychometricFnsAdjustedByConf2  = zeros(100, 2, 21); % HC, LC

% Bayesian
confFns                  = zeros(100, 21);
confFnsV1Damaged         = zeros(100, 21);
confFnsV1Adjusted        = zeros(100, 21);

% SDT
confFns2                  = zeros(100, 21);
confFnsV1Damaged2         = zeros(100, 21);
confFnsV1Adjusted2        = zeros(100, 21);


for run_itr = 1:100
% Each itr is evaluating probably a different session

disp(run_itr)

% ----------------------------------
% Parameters
% ----------------------------------
nNeurons = 100;        % Number of neurons
stimDuration = 1;      % Stimulus duration in seconds
varGain = 0.5;         % Variance in gain for modulated Poisson process
timeStep = 0.001;      % Time step (1ms) for binning the stimulus duration
propDamaged = 0.8;     % Proportion of damaged neurons

% Damaged neurons indexes
nrnCntDamaged = floor( propDamaged*nNeurons );
damagedNrnIdxes = randperm(nNeurons, nrnCntDamaged);
assert( numel(unique(damagedNrnIdxes(:))) == numel(damagedNrnIdxes) ); % Must be non repeating integers
intactNrnIdxes = setdiff(1:nNeurons, damagedNrnIdxes);

% Stimulus parameters (angles in radians)
stimParam.startInterval = deg2rad(90 - 21);              % Start of stimulus interval (radians)
stimParam.endInterval = deg2rad(90 + 21);                % End of stimulus interval (radians)
stimParam.numStim = 21;                                % Number of unique stimuli
stimParam.countPerStim = 200;                          % Number of trials per stimulus
ntrials = stimParam.numStim * stimParam.countPerStim;  % Total number of trials

% Add random noise to the stimulus orientation
stimNoise = 0 + 0.1 * randn(1, ntrials); % What is std dev here? 5.73 degrees
stimVector = repelem(linspace(stimParam.startInterval, stimParam.endInterval, ...
    stimParam.numStim), stimParam.countPerStim);  % Vector of stimuli
noisyStimVector = stimVector + stimNoise;         % Noisy stimulus vector

% Shuffle stim vector - this is important for studying effect of top down
% effect
shuffleIdx = randperm(ntrials);
stimVector = stimVector(shuffleIdx);
noisyStimVector = noisyStimVector(shuffleIdx); 

% Neuron preferred orientations and time bins
neuronsPrefOrientation = zeros(1, nNeurons);      % Preferred orientation for each neuron
timeBins = 0:timeStep:stimDuration;               % Time bins from 0 to stimDuration

% Stimulus response profile based on a normal distribution
stimRespProfile = 1 + zeros(1, numel(timeBins));
gainVector = 1 + zeros(ntrials, nNeurons); % constant gain - NO gain modulation

% Neuron tuning parameters for three conditions
tuningParams           = getTuningParams(nNeurons);
tuningParamsV1Damaged  = getTuningParamsV1Damaged(tuningParams, damagedNrnIdxes);
tuningParamsAdjusted   = getTuningParamsAdjusted(tuningParams, intactNrnIdxes);

% Assign random preferred orientations to the neurons
neuronsPrefOrientation(:) = pi * rand(1, nNeurons);  % Random orientations from -pi to pi

% Structures to store final neuron spikes
% Preallocate result and spike response matrices
trialDecisions          = zeros(1, ntrials);

% Bayesian
trlConfVars               = zeros(1, ntrials);
trlConfVarsV1Damaged      = zeros(1, ntrials);
trlConfVarsV1Ajusted      = zeros(1, ntrials);

trlConfReports            = zeros(1, ntrials);
trlConfReportsV1Damaged   = zeros(1, ntrials);
trlConfReportsV1Adjusted  = zeros(1, ntrials);

% SDT
trlConfVars2            = zeros(1, ntrials);
trlConfVarsV1Damaged2   = zeros(1, ntrials);
trlConfVarsV1Ajusted2   = zeros(1, ntrials);

trlConfReports2            = zeros(1, ntrials);
trlConfReportsV1Damaged2   = zeros(1, ntrials);
trlConfReportsV1Adjusted2  = zeros(1, ntrials);

decodedPDFs             = cell(ntrials, 1);
decodedPDFsV1Damaged    = cell(ntrials, 1);
decodedPDFsV1Adjusted   = cell(ntrials, 1);

neuronSpikeResponses = false(ntrials, nNeurons, length(timeBins)); % Creating a logical matrix to save memory

trialDecisionsDamaged = zeros(1, ntrials);
neuronSpikeResponsesDamaged = false(ntrials, nNeurons, length(timeBins)); % Creating a logical matrix to save memory

trialDecisionsAdjusted = zeros(1, ntrials);
neuronSpikeResponsesAdjusted = false(ntrials, nNeurons - nrnCntDamaged, length(timeBins)); % Creating a logical matrix to save memory

% Plot orientation tuning of damaged and intact neurons
stimVector_tuningFn = linspace(0, pi, 200);
tuningFns = orientationTunedFiringRate(stimVector_tuningFn, ...
    neuronsPrefOrientation, tuningParams);
tuningFns = tuningFns';

tuningFnsV1Damaged = orientationTunedFiringRate(stimVector_tuningFn, ...
    neuronsPrefOrientation, tuningParamsV1Damaged);
tuningFnsV1Damaged = tuningFnsV1Damaged';

tuningFnsAdjusted = orientationTunedFiringRate(stimVector_tuningFn, ...
    neuronsPrefOrientation(intactNrnIdxes), tuningParamsAdjusted);
tuningFnsAdjusted = tuningFnsAdjusted';

% ----------------------------------
% Computing stimulus response begins
% ----------------------------------

% STEP 1: Compute orientation-tuned firing rates for each trial
% Output: 
%  - firingRates: matrix of firing rates (nTrials x nNeurons)
firingRates = orientationTunedFiringRate(noisyStimVector, ...
    neuronsPrefOrientation, tuningParams);

firingRatesV1Damaged = orientationTunedFiringRate(noisyStimVector, ...
    neuronsPrefOrientation, tuningParamsV1Damaged);

firingRatesAdjusted = orientationTunedFiringRate(noisyStimVector, ...
    neuronsPrefOrientation(intactNrnIdxes), tuningParamsAdjusted);


for trialIDx = 1:ntrials
    % if mod(trialIDx, stimParam.countPerStim) == 0
    %     disp(trialIDx)
    % end

    [~, thetaMLE, decision, pdf] = decodeDecision(gainVector, trialIDx, timeBins, firingRates, ...
        stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, tuningParams);
    
    % Store spike trains and decision results
    % neuronSpikeResponses(trialIDx, :, :) = logical(spikes);  % Store spikes
    trialDecisions(trialIDx) = decision;  % Store decision result (CW or CCW)
    decodedPDFs{trialIDx}    = pdf;       % save pdf

    % compute confidence
    d_criteria               = 90;
    [probCorrect, conf]      = computeConfidence(decision, d_criteria, pdf.pdf, pdf.x_deg);
    trlConfVars(trialIDx)    = probCorrect;
    trlConfReports(trialIDx) = conf;
    
    [confVar, conf]           = computeConfidenceSDT(rad2deg(thetaMLE), d_criteria, pdf.sigma);
    trlConfVars2(trialIDx)    = confVar;
    trlConfReports2(trialIDx) = conf;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V1 damaged
    % Tuning function used for decoding should be same as
    % the previous one. The tuning function used for generating spikes
    % should be the damaged one.
    % Spikes are generated as per damaged neurons whereas decoding is done
    % as per original tuning function.
    % [spikes, decision] = decodeDecision(gainVector, trialIDx, timeBins, firingRatesV1Damaged, ...
    %     stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, tuningParamsV1Damaged);
    [~, thetaMLE, decision, pdf] = decodeDecision(gainVector, trialIDx, timeBins, ...
        firingRatesV1Damaged, ... % Firing rates for damaged population
        stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, ...
        tuningParams); % Orignal tuning function
    

    % Store spike trains and decision results
    % neuronSpikeResponsesDamaged(trialIDx, :, :) = logical(spikes);  % Store spikes
    trialDecisionsDamaged(trialIDx) = decision;  % Store decision result (CW or CCW)
    decodedPDFsV1Damaged{trialIDx}  = pdf;

    % compute confidence
    d_criteria                        = 90;
    [probCorrect, conf]               = computeConfidence(decision, d_criteria, pdf.pdf, pdf.x_deg);
    trlConfVarsV1Damaged(trialIDx)    = probCorrect;
    trlConfReportsV1Damaged(trialIDx) = conf;

    [confVar, conf]                    = computeConfidenceSDT(rad2deg(thetaMLE), d_criteria, pdf.sigma);
    trlConfVarsV1Damaged2(trialIDx)    = confVar;
    trlConfReportsV1Damaged2(trialIDx) = conf;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Adjusted decoder
    % Maybe use the same spike train as previous spikes obtained
    [~, thetaMLE, decision, pdf] = decodeDecision( ...
        gainVector(:, intactNrnIdxes), ...
        trialIDx, timeBins, firingRatesAdjusted, ...
        stimRespProfile, timeStep, numel(intactNrnIdxes), stimDuration, ...
        neuronsPrefOrientation(intactNrnIdxes), tuningParamsAdjusted);
    
    % Store spike trains and decision results
    % neuronSpikeResponsesAdjusted(trialIDx, :, :) = logical(spikes);  % Store spikes
    trialDecisionsAdjusted(trialIDx) = decision;  % Store decision result (CW or CCW)
    decodedPDFsV1Adjusted{trialIDx}  = pdf; 

    % compute confidence
    d_criteria                         = 90;
    [probCorrect, conf]                = computeConfidence(decision, d_criteria, pdf.pdf, pdf.x_deg);
    trlConfVarsV1Ajusted(trialIDx)     = probCorrect;
    trlConfReportsV1Adjusted(trialIDx) = conf;

    [confVar, conf]                     = computeConfidenceSDT(rad2deg(thetaMLE), d_criteria, pdf.sigma);
    trlConfVarsV1Ajusted2(trialIDx)     = confVar;
    trlConfReportsV1Adjusted2(trialIDx) = conf;
end


uniqStim = unique(stimVector);

psychometricFn          = zeros(numel(uniqStim), 1);
psychometricFnDamaged   = zeros(numel(uniqStim), 1);
psychometricFnAdjusted  = zeros(numel(uniqStim), 1);

psycFnByConf2           = zeros(2, numel(uniqStim)); % HC, LC
psychFnDamagedByConf2   = zeros(2, numel(uniqStim)); % HC, LC
psychFnAdjustedByConf2  = zeros(2, numel(uniqStim)); % HC, LC

% Bayesian
confFn                  = zeros(numel(uniqStim), 1);
confFnV1Damaged         = zeros(numel(uniqStim), 1);
confFnV1Adjusted        = zeros(numel(uniqStim), 1);

% SDT
confFn2                  = zeros(numel(uniqStim), 1);
confFnV1Damaged2         = zeros(numel(uniqStim), 1);
confFnV1Adjusted2        = zeros(numel(uniqStim), 1);

for i=1:stimParam.numStim
    stimOrientation = uniqStim(i);
    givenOrientationTrialIDxes = find(stimVector == stimOrientation);
    
    % Intact V1
    decision = trialDecisions(givenOrientationTrialIDxes);
    confReports2 = trlConfReports2(givenOrientationTrialIDxes);
    
    percent_CCW = length(find(decision == -1)) / numel(decision);
    psychometricFn(i) = percent_CCW; % This is actually CCW but not a big deal
    
    percent_CCW_HC = length(find(decision == -1 & confReports2 == 1)) / numel(decision(confReports2 == 1));
    percent_CCW_LC = length(find(decision == -1 & confReports2 == 0)) / numel(decision(confReports2 == 0));
    psycFnByConf2(1, i) = percent_CCW_HC;
    psycFnByConf2(2, i) = percent_CCW_LC;

    propHC      = sum(trlConfReports(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFn(i)   = propHC;
    propHC      = sum(trlConfReports2(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFn2(i)  = propHC;

    % Damaged V1
    decision = trialDecisionsDamaged(givenOrientationTrialIDxes);
    confReports2 = trlConfReportsV1Damaged2(givenOrientationTrialIDxes);
    
    percent_CCW = length(find(decision == -1)) / numel(decision);
    psychometricFnDamaged(i) = percent_CCW;
    
    percent_CCW_HC = length(find(decision == -1 & confReports2 == 1)) / numel(decision(confReports2 == 1));
    percent_CCW_LC = length(find(decision == -1 & confReports2 == 0)) / numel(decision(confReports2 == 0));
    psychFnDamagedByConf2(1, i) = percent_CCW_HC;
    psychFnDamagedByConf2(2, i) = percent_CCW_LC;

    propHC               = sum(trlConfReportsV1Damaged(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnV1Damaged(i)   = propHC;
    propHC               = sum(trlConfReportsV1Damaged2(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnV1Damaged2(i)  = propHC;

    % Adjusted V1
    decision = trialDecisionsAdjusted(givenOrientationTrialIDxes);
    confReports2 = trlConfReportsV1Adjusted2(givenOrientationTrialIDxes);
    
    percent_CCW = length(find(decision == -1)) / numel(decision);
    psychometricFnAdjusted(i) = percent_CCW;
    
    percent_CCW_HC = length(find(decision == -1 & confReports2 == 1)) / numel(decision(confReports2 == 1));
    percent_CCW_LC = length(find(decision == -1 & confReports2 == 0)) / numel(decision(confReports2 == 0));
    psychFnAdjustedByConf2(1, i) = percent_CCW_HC;
    psychFnAdjustedByConf2(2, i) = percent_CCW_LC;
    
    propHC                = sum(trlConfReportsV1Adjusted(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnV1Adjusted(i)   = propHC;
    propHC                = sum(trlConfReportsV1Adjusted2(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnV1Adjusted2(i)  = propHC;

end

psychometricFns(run_itr, :)          = psychometricFn';
psychometricFnsDamaged(run_itr, :)   = psychometricFnDamaged';
psychometricFnsAdjusted(run_itr, :)  = psychometricFnAdjusted';

% Bayesian
confFns(run_itr, :)    = confFn';           
confFnsV1Damaged(run_itr, :) = confFnV1Damaged';
confFnsV1Adjusted(run_itr, :) = confFnV1Adjusted';

% SDT
confFns2(run_itr, :) = confFn2';
confFnsV1Damaged2(run_itr, :) = confFnV1Damaged2';
confFnsV1Adjusted2 (run_itr, :) = confFnV1Adjusted2';

psychometricFnsByConf2(run_itr, :, :)          = psycFnByConf2;
psychometricFnsDamagedByConf2(run_itr, :, :)   = psychFnDamagedByConf2;
psychometricFnsAdjustedByConf2(run_itr, :, :)  = psychFnAdjustedByConf2;


end

data.psychometricFns                = psychometricFns;
data.psychometricFnsDamaged         = psychometricFnsDamaged;
data.psychometricFnsAdjusted        = psychometricFnsAdjusted;

% Bayesian 
data.confFns           = confFns;
data.confFnsV1Damaged  = confFnsV1Damaged;
data.confFnsV1Adjusted = confFnsV1Adjusted;

% SDT
data.confFns2           = confFns2;
data.confFnsV1Damaged2  = confFnsV1Damaged2;
data.confFnsV1Adjusted2 = confFnsV1Adjusted2;

data.psychometricFnsByConf2         = psychometricFnsByConf2;
data.psychometricFnsDamagedByConf2  = psychometricFnsDamagedByConf2;
data.psychometricFnsAdjustedByConf2 = psychometricFnsAdjustedByConf2;


save("damagedV1_Conf_sim.mat", "data")


%% Plot results
close all

figure

for i = 1:100
    
    subplot(2, 3, 1)
    hold on
    plot(rad2deg(uniqStim), psychometricFns(i, :), 'DisplayName', "Intact V1", 'Color', [0 0.7 0 0.2]); 
    plot(rad2deg(uniqStim), psychometricFnsDamaged(i, :), 'DisplayName', "Damaged V1", 'Color',[1 0 0 0.2]); 
    plot(rad2deg(uniqStim), psychometricFnsAdjusted(i, :), 'DisplayName', "Adjusted V1", 'Color',[0 0 1 0.2]); 
    
    h1 = plot(rad2deg(uniqStim), mean( psychometricFns, 1), DisplayName="Intact V1", Color=[0 0.7 0], LineWidth=1.5);
    h2 = plot(rad2deg(uniqStim), mean( psychometricFnsDamaged, 1), DisplayName="Damaged V1", Color="red", LineWidth=1.5);
    h3 = plot(rad2deg(uniqStim), mean( psychometricFnsAdjusted, 1), DisplayName="Adjusted V1", Color="blue", LineWidth=1.5);

    hold off
    xlabel("Orientation")
    ylabel("prop CCW")
    % legend
    
    subplot(2, 3, 2)
    hold on
    plot(rad2deg(uniqStim), confFns(i, :), 'DisplayName', "Intact V1", 'Color', [0 0.7 0 0.2]); 
    plot(rad2deg(uniqStim), confFnsV1Damaged(i, :), 'DisplayName', "Damaged V1", 'Color',[1 0 0 0.2]); 
    plot(rad2deg(uniqStim), confFnsV1Adjusted(i, :), 'DisplayName', "Adjusted V1", 'Color',[0 0 1 0.2]);  
    
    h1_ = plot(rad2deg(uniqStim), mean( confFns, 1), DisplayName="Intact V1", Color=[0 0.7 0], LineWidth=1.5);
    h2_ = plot(rad2deg(uniqStim), mean( confFnsV1Damaged, 1), DisplayName="Damaged V1", Color="red", LineWidth=1.5);
    h3_ = plot(rad2deg(uniqStim), mean( confFnsV1Adjusted, 1), DisplayName="Adjusted V1", Color="blue", LineWidth=1.5);
    
    hold off
    xlabel("Orientation")
    ylabel("prop HC")
    title("Bayesian")
    % legend
    
    subplot(2, 3, 3)
    hold on
    plot(rad2deg(uniqStim), confFns2(i, :), 'DisplayName', "Intact V1", 'Color', [0 0.7 0 0.2]); 
    plot(rad2deg(uniqStim), confFnsV1Damaged2(i, :), 'DisplayName', "Damaged V1", 'Color',[1 0 0 0.2]); 
    plot(rad2deg(uniqStim), confFnsV1Adjusted2(i, :), 'DisplayName', "Adjusted V1", 'Color',[0 0 1 0.2]); 
    
    h1__ = plot(rad2deg(uniqStim), mean(confFns2, 1), DisplayName="Intact V1", Color=[0 0.7 0], LineWidth=1.5);
    h2__ = plot(rad2deg(uniqStim), mean(confFnsV1Damaged2, 1), DisplayName="Damaged V1", Color="red", LineWidth=1.5);
    h3__ = plot(rad2deg(uniqStim), mean(confFnsV1Adjusted2, 1), DisplayName="Adjusted V1", Color="blue", LineWidth=1.5);
    
    hold off
    xlabel("Orientation")
    ylabel("prop HC")
    title("SDT")
    % legend

    subplot(2, 3, 4)
    hold on
    plot(rad2deg(uniqStim), squeeze( psychometricFnsByConf2(i, 1, :) ), 'DisplayName', "HC", 'Color', [0 0.7 0 0.2]); 
    plot(rad2deg(uniqStim), squeeze( psychometricFnsByConf2(i, 2, :) ), 'DisplayName', "LC", 'Color',[1 0 0 0.2]); 
    
    d = squeeze(psychometricFnsByConf2(:,1,:));
    hc1 = plot(rad2deg(uniqStim), mean( d , 1, 'omitnan'), DisplayName="HC", LineWidth=1.5, Color=[0 0.7 0]);
    d = squeeze(psychometricFnsByConf2(:,2,:));
    lc1 = plot(rad2deg(uniqStim), mean( d, 1, 'omitnan'), DisplayName="LC", LineWidth=1.5, Color=[1 0 0]);
    
    hold off
    xlabel("Orientation")
    title("Intact V1")
    ylabel("prop CCW")

    subplot(2, 3, 5)
    hold on
    plot(rad2deg(uniqStim), squeeze( psychometricFnsDamagedByConf2(i, 1, :) ), 'DisplayName', "HC", 'Color', [0 0.7 0 0.2]); 
    plot(rad2deg(uniqStim), squeeze( psychometricFnsDamagedByConf2(i, 2, :) ), 'DisplayName', "LC", 'Color',[1 0 0 0.2]); 
    
    d = squeeze(psychometricFnsDamagedByConf2(:,1,:));
    hc2 = plot(rad2deg(uniqStim), mean(d , 1, 'omitnan'), DisplayName="HC", LineWidth=1.5, Color=[0 0.7 0]);
    d = squeeze(psychometricFnsDamagedByConf2(:,2,:));
    lc2 = plot(rad2deg(uniqStim), mean( d, 1, 'omitnan'), DisplayName="LC", LineWidth=1.5, Color=[1 0 0]);
    
    hold off
    xlabel("Orientation")
    title("Damaged V1")
    ylabel("prop CCW")
    
    subplot(2, 3, 6)
    hold on
    plot(rad2deg(uniqStim), squeeze( psychometricFnsAdjustedByConf2(i, 1, :) ), 'DisplayName', "HC", 'Color', [0 0.7 0 0.2]); 
    plot(rad2deg(uniqStim), squeeze( psychometricFnsAdjustedByConf2(i, 2, :) ), 'DisplayName', "LC", 'Color', [1 0 0 0.2]); 
    
    d = squeeze(psychometricFnsAdjustedByConf2(:,1,:));
    hc3 = plot(rad2deg(uniqStim), mean(d , 1, 'omitnan'), DisplayName="HC", LineWidth=1.5, Color=[0 0.7 0]);
    d = squeeze(psychometricFnsAdjustedByConf2(:,2,:));
    lc3 = plot(rad2deg(uniqStim), mean( d, 1, 'omitnan'), DisplayName="LC", LineWidth=1.5, Color=[1 0 0]);
    
    hold off
    xlabel("Orientation")
    title("Adjusted V1")
    ylabel("prop CCW")

end

legend([h1 h2 h3], ...
       {'Intact V1','Damaged V1','Adjusted V1'}, ...
       'Location','best');

legend([h1_ h2_ h3_], ...
       {'Intact V1','Damaged V1','Adjusted V1'}, ...
       'Location','best');

legend([h1__ h2__ h3__], ...
       {'Intact V1','Damaged V1','Adjusted V1'}, ...
       'Location','best');

legend([hc1 lc1], ...
       {'HC','LC'}, ...
       'Location','best');

legend([hc2 lc2], ...
       {'HC','LC'}, ...
       'Location','best');

legend([hc2 lc2], ...
       {'HC','LC'}, ...
       'Location','best');


% %%
% 
% % Plot PDFs for a particular orientation to see if the decoded likelihood
% % changes trial to trial for the same orientation
% uniqStimVals = unique(stimVector);
% fltTrlIdx    = (stimVector == deg2rad(100.5));
% trlVals      = 1:ntrials;
% fltTrialVals = trlVals(fltTrlIdx);
% 
% figure
% 
% subplot(4, 3, 1)
% hold on
% 
% stdVals = zeros(numel(fltTrialVals), 1);
% for j=1:numel(fltTrialVals)
%     i = fltTrialVals(j);
%     pdf = decodedPDFs{i}.pdf;
%     x   = decodedPDFs{i}.x_deg;
%     stdVals(j) = decodedPDFs{i}.sigma;
% 
%     plot(x, pdf)
% end
% xlabel("orientation")
% ylabel("PDF")
% title("Intact V1")
% % xlim([60, 120])
% hold off
% 
% subplot(4, 3, 4)
% histogram(stdVals)
% title("Std vals")
% 
% subplot(4, 3, 7)
% histogram(trlConfVars(fltTrlIdx))
% title("Conf var")
% 
% subplot(4, 3, 10)
% histogram(trlConfVars2(fltTrlIdx))
% title("Conf var")
% 
% subplot(4, 3, 2)
% hold on
% 
% stdVals = zeros(numel(fltTrialVals), 1);
% for j=1:numel(fltTrialVals)
%     i = fltTrialVals(j);
%     pdf = decodedPDFsV1Damaged{i}.pdf;
%     x   = decodedPDFsV1Damaged{i}.x_deg;
%     stdVals(j) = decodedPDFsV1Damaged{i}.sigma;
%     
%     plot(x, pdf)
% end
% xlabel("orientation")
% ylabel("PDF")
% title("Damaged V1")
% hold off
% 
% subplot(4, 3, 5)
% histogram(stdVals)
% title("Std vals")
% 
% subplot(4, 3, 8)
% histogram(trlConfVarsV1Damaged(fltTrlIdx))
% title("Conf var")
% 
% subplot(4, 3, 11)
% histogram(trlConfVarsV1Damaged2(fltTrlIdx))
% title("Conf var")
% 
% subplot(4, 3, 3)
% hold on
% 
% stdVals = zeros(numel(fltTrialVals), 1);
% for j=1:numel(fltTrialVals)
%     i = fltTrialVals(j);
%     pdf = decodedPDFsV1Adjusted{i}.pdf;
%     x   = decodedPDFsV1Adjusted{i}.x_deg;
%     stdVals(j) = decodedPDFsV1Adjusted{i}.sigma;
%     
%     plot(x, pdf)
% end
% xlabel("orientation")
% ylabel("PDF")
% title("Adjusted V1")
% hold off
% 
% subplot(4, 3, 6)
% histogram(stdVals)
% title("Std vals")
% 
% subplot(4, 3, 9)
% histogram(trlConfVarsV1Ajusted(fltTrlIdx))
% title("Conf var")
% 
% subplot(4, 3, 12)
% histogram(trlConfVarsV1Ajusted2(fltTrlIdx))
% title("Conf var")

% Fix PDF first
% Quantify confidence - Probability of being correct - choose decision -
% calculate probability - choose a confidence criteria 
% Other option to quantify confidence - get standard deviation of the PDF -
% compute confidence variable from the std val - use criteria to compute
% confidence. Things to check: does std changes from trial to trial or is it fixed (Look at the distribution)?

%%
function [tuningParams] = getTuningParams(nNeurons)

% Neuron tuning parameters
tuningParams.d = zeros(1, nNeurons) + 0;           % Direction selectivity (fixed at 0)
tuningParams.alpha = zeros(1, nNeurons) + 2;       % Aspect ratio (fixed at 2)
tuningParams.b = zeros(1, nNeurons) + 2;           % Controls sharpness of tuning curve (fixed at 2)
tuningParams.q = zeros(1, nNeurons) + 1;           % Controls amplitude of peak firing rate (variable)
tuningParams.w = zeros(1, nNeurons) + 1;           % Unused parameter (set to 1)
tuningParams.UNTUNED_FILTER_AMPL = 0;              % Untuned filter amplitude (fixed at 0)
tuningParams.eps1 = zeros(1, nNeurons);            % Controls dynamic range (variable)
tuningParams.beta = zeros(1, nNeurons) + 1;        % Controls dynamic range (variable)

% Overriding certain tuning parameters with randomized values
tuningParams.q(:) = 8 * rand(1, nNeurons);               % Random values for peak firing rate control
tuningParams.beta(:) = lognrnd(2.5, 0.5, 1, nNeurons);   % Random values for beta parameter - evoked activity
tuningParams.eps1(:) = lognrnd(1, 0.8, 1, nNeurons);     % Random values for dynamic range control - spontaneous activity

end

function [tuningParamsV1Damaged] = getTuningParamsV1Damaged(tuningParams, damagedNrnIdxes)

% For damaged model, evoked activity for damaged neurons is set to zero.

% Neuron tuning parameters
tuningParamsV1Damaged.d = tuningParams.d;                      % Direction selectivity (fixed at 0)
tuningParamsV1Damaged.alpha = tuningParams.alpha;              % Aspect ratio (fixed at 2)
tuningParamsV1Damaged.b = tuningParams.b;                      % Controls sharpness of tuning curve (fixed at 2)
tuningParamsV1Damaged.q = tuningParams.q;                      % Controls amplitude of peak firing rate (variable)
tuningParamsV1Damaged.w = tuningParams.w;                      % Unused parameter (set to 1)
tuningParamsV1Damaged.UNTUNED_FILTER_AMPL = tuningParams.UNTUNED_FILTER_AMPL;  % Untuned filter amplitude (fixed at 0)
tuningParamsV1Damaged.eps1 = tuningParams.eps1;                % Controls dynamic range (variable) - also sets baseline firing rate (spontaneous maybe)
tuningParamsV1Damaged.beta = tuningParams.beta;                % Controls dynamic range (variable)

% Set evoked activity zero for damaged neurons
tuningParamsV1Damaged.beta(damagedNrnIdxes) = 0;               % Evoked activity set to zero for damaged neuron

end

function [tuningParamsAdjusted] = getTuningParamsAdjusted(tuningParams, intactNrnIdxes)

% Remove damaged neurons altogether

% Tuning params just for intact neurons
tuningParamsAdjusted.d = tuningParams.d(intactNrnIdxes);                      % Direction selectivity (fixed at 0)
tuningParamsAdjusted.alpha = tuningParams.alpha(intactNrnIdxes);              % Aspect ratio (fixed at 2)
tuningParamsAdjusted.b = tuningParams.b(intactNrnIdxes);                      % Controls sharpness of tuning curve (fixed at 2)
tuningParamsAdjusted.q = tuningParams.q(intactNrnIdxes);                      % Controls amplitude of peak firing rate (variable)
tuningParamsAdjusted.w = tuningParams.w(intactNrnIdxes);                      % Unused parameter (set to 1)
tuningParamsAdjusted.UNTUNED_FILTER_AMPL = tuningParams.UNTUNED_FILTER_AMPL;  % Untuned filter amplitude (fixed at 0)
tuningParamsAdjusted.eps1 = tuningParams.eps1(intactNrnIdxes);                % Controls dynamic range (variable)
tuningParamsAdjusted.beta = tuningParams.beta(intactNrnIdxes);                % Controls dynamic range (variable)

end

function [spikes, thetaMLE, decision, pdf] = decodeDecision(gainVector, trialIDx, timeBins, firingRates, ...
    stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, tuningParams)

% Modulate gain of current trial based on previous trial
% nNeurons x No time bins
trlGainVector = squeeze(repmat(gainVector(trialIDx, :), [1, 1, length(timeBins)])); % Extract gain vector for this trial for each timebin

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Intact
% STEP 2: Compute stimulus response for each neuron over time
% This multiplies the firing rates with a time-dependent stimulus response profile
% Output: 
%  - trlStimResponse: response of each neuron over time for each trial (nNeurons x nTimeBins)
trlStimResponse = firingRates(trialIDx, :)'.*stimRespProfile;

% STEP 3: Generate modulated Poisson spikes for each trial
% Output:
%  - spikes: spike trains for each neuron in the trial
%  - modStimResponse: modified stimulus response after gain modulation
params = struct();
params.timeStep = timeStep;
params.timeBins = timeBins;
params.nNeurons = nNeurons;
[spikes, ~] = generateModulatedPoissonSpikes(trlStimResponse, ...
    trlGainVector, params);

% STEP 4: Decode the stimulus orientation based on the spike trains
% Output:
%  - thetaMLE: maximum likelihood estimate of stimulus orientation based on spikes
%  - decodingError: error between decoded orientation and actual stimulus
params.stimDuration = stimDuration;
[thetaMLE, pdf] = decodeOrientationFromSpikes(spikes, ...
    neuronsPrefOrientation, params, tuningParams);
% decodingError = thetaMLE - noisyStimVector(trialIDx);

% Decision: CCW (-1) or CW (1) based on decoded orientation
decision = (thetaMLE > pi/2)*(-1) + (thetaMLE < pi/2)*(1); % (thetaMLE <= pi/2)*(1)

end


function [probCorrect, conf] = computeConfidence(decision, d_criteria, pdf, x_deg)
% Everything in degrees here
% Decision: CCW (-1) or CW (1) based on decoded orientation

if decision == -1 % CCW
    fltIdx = x_deg > d_criteria;
else % CW
    fltIdx = x_deg < d_criteria;
end

fltPdf = pdf(fltIdx);
fltX   = x_deg(fltIdx);
dx = fltX(2) - fltX(1);

probCorrect = sum( fltPdf.*dx );

c_criteria = 0.9;
conf = probCorrect > c_criteria;

% TODO: later apply some confidence criteria to categorize it into high and
% low confidence
end

function [confVar, conf] = computeConfidenceSDT(thetaMLE, d_criteria, sigma)
% Everything in degrees here

Vc = abs(thetaMLE - d_criteria) / sigma;
confVar = Vc;

c_criteria = 1.5;
conf = confVar > c_criteria;

% TODO: later apply some confidence criteria to categorize it into high and
% low confidence
end