%% 
clear all
close all
% clc

addpath('/Users/avinashranjan/Desktop/UT Austin/Goris lab/Model_V1_damage/V1_DamageModel/Model/Scripts/')

% Does confidence criteria changes after damage?
% Add gain variability
% TODO: run many iterations

rng('shuffle'); % Ensures that different random numbers are generated every time
      
% TIP: Keep all possible 1D variables as row vectors.
% NOTE: Time step should be small relative to the spike rate of a single neuron,
% so that the probability of firing doesn't shoot up. 

% TODO: Add a check to ensure that the firing rate is not too large 
% compared to the time window.

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
stimParam.numStim = 101;  %21                               % Number of unique stimuli
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


%% Prepare data structures

uniqStim = unique(stimVector);

psychometricFns          = zeros(numel(uniqStim), 1);
psychometricFnsDamaged   = zeros(numel(uniqStim), 1);
psychometricFnsAdjusted  = zeros(numel(uniqStim), 1);

psychometricFnsByConf2          = zeros(2, numel(uniqStim)); % HC, LC
psychometricFnsDamagedByConf2   = zeros(2, numel(uniqStim)); % HC, LC
psychometricFnsAdjustedByConf2  = zeros(2, numel(uniqStim)); % HC, LC

% TODO: just use one - this will becoem super confusing
% Bayesian
confFns                  = zeros(numel(uniqStim), 1);
confFnsV1Damaged         = zeros(numel(uniqStim), 1);
confFnsV1Adjusted        = zeros(numel(uniqStim), 1);

% SDT
confFns2                  = zeros(numel(uniqStim), 1);
confFnsV1Damaged2         = zeros(numel(uniqStim), 1);
confFnsV1Adjusted2        = zeros(numel(uniqStim), 1);

% Choice counts: Order - (CW, HC), (CW, LC), (CCW, HC), (CCW, LC)
nChoicesV1Intact2   = zeros(4, numel(uniqStim));
nChoicesV1Damaged2  = zeros(4, numel(uniqStim));
nChoicesV1Adjusted2 = zeros(4, numel(uniqStim));


for i=1:stimParam.numStim
    stimOrientation = uniqStim(i);
    givenOrientationTrialIDxes = find(stimVector == stimOrientation);
    
    % Intact V1
    decision = trialDecisions(givenOrientationTrialIDxes);
    confReports = trlConfReports(givenOrientationTrialIDxes);
    confReports2 = trlConfReports2(givenOrientationTrialIDxes);
    
    percent_CCW = length(find(decision == -1)) / numel(decision);
    psychometricFns(i) = percent_CCW; % This is actually CCW but not a big deal
    
    percent_CCW_HC = length(find(decision == -1 & confReports2 == 1)) / numel(decision(confReports2 == 1));
    percent_CCW_LC = length(find(decision == -1 & confReports2 == 0)) / numel(decision(confReports2 == 0));
    psychometricFnsByConf2(1, i) = percent_CCW_HC;
    psychometricFnsByConf2(2, i) = percent_CCW_LC;
    
    propHC      = sum(trlConfReports(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFns(i)  = propHC;
    propHC      = sum(trlConfReports2(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFns2(i) = propHC;
    
    % choice count
    nChoicesV1Intact2(1, i) = sum( ( (decision == 1) & (confReports2 == 1) ) );  %(CW, HC)
    nChoicesV1Intact2(2, i) = sum( ( (decision == 1) & (confReports2 == 0) ) );  %(CW, LC)
    nChoicesV1Intact2(3, i) = sum( ( (decision == -1) & (confReports2 == 1) ) ); %(CCW, HC)
    nChoicesV1Intact2(4, i) = sum( ( (decision == -1) & (confReports2 == 0) ) ); %(CCW, LC)
    
    % Damaged V1
    decision = trialDecisionsDamaged(givenOrientationTrialIDxes);
    confReports = trlConfReportsV1Damaged(givenOrientationTrialIDxes);
    confReports2 = trlConfReportsV1Damaged2(givenOrientationTrialIDxes);
    
    percent_CCW = length(find(decision == -1)) / numel(decision);
    psychometricFnsDamaged(i) = percent_CCW;
    
    percent_CCW_HC = length(find(decision == -1 & confReports2 == 1)) / numel(decision(confReports2 == 1));
    percent_CCW_LC = length(find(decision == -1 & confReports2 == 0)) / numel(decision(confReports2 == 0));
    psychometricFnsDamagedByConf2(1, i) = percent_CCW_HC;
    psychometricFnsDamagedByConf2(2, i) = percent_CCW_LC;

    propHC               = sum(trlConfReportsV1Damaged(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnsV1Damaged(i)  = propHC;
    propHC               = sum(trlConfReportsV1Damaged2(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnsV1Damaged2(i) = propHC;

    % choice count
    nChoicesV1Damaged2(1, i) = sum( ( (decision == 1) & (confReports2 == 1) ) );  %(CW, HC)
    nChoicesV1Damaged2(2, i) = sum( ( (decision == 1) & (confReports2 == 0) ) );  %(CW, LC)
    nChoicesV1Damaged2(3, i) = sum( ( (decision == -1) & (confReports2 == 1) ) ); %(CCW, HC)
    nChoicesV1Damaged2(4, i) = sum( ( (decision == -1) & (confReports2 == 0) ) ); %(CCW, LC)
    
    % Adjusted V1
    decision = trialDecisionsAdjusted(givenOrientationTrialIDxes);
    confReports = trlConfReportsV1Adjusted(givenOrientationTrialIDxes);
    confReports2 = trlConfReportsV1Adjusted2(givenOrientationTrialIDxes);
    
    percent_CCW = length(find(decision == -1)) / numel(decision);
    psychometricFnsAdjusted(i) = percent_CCW;

    propHC                = sum(trlConfReportsV1Adjusted(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnsV1Adjusted(i)  = propHC;
    propHC                = sum(trlConfReportsV1Adjusted2(givenOrientationTrialIDxes) == 1) / numel(decision);
    confFnsV1Adjusted2(i) = propHC;
    
    percent_CCW_HC = length(find(decision == -1 & confReports2 == 1)) / numel(decision(confReports2 == 1));
    percent_CCW_LC = length(find(decision == -1 & confReports2 == 0)) / numel(decision(confReports2 == 0));
    psychometricFnsAdjustedByConf2(1, i) = percent_CCW_HC;
    psychometricFnsAdjustedByConf2(2, i) = percent_CCW_LC;

    % choice count
    nChoicesV1Adjusted2(1, i) = sum( ( (decision == 1) & (confReports2 == 1) ) );  %(CW, HC)
    nChoicesV1Adjusted2(2, i) = sum( ( (decision == 1) & (confReports2 == 0) ) );  %(CW, LC)
    nChoicesV1Adjusted2(3, i) = sum( ( (decision == -1) & (confReports2 == 1) ) ); %(CCW, HC)
    nChoicesV1Adjusted2(4, i) = sum( ( (decision == -1) & (confReports2 == 0) ) ); %(CCW, LC)
    
end

%% Plot psychometric curves
figure

subplot(2, 3, 1)
hold on
plot(uniqStim, nChoicesV1Intact2(1, :), DisplayName='CW, HC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Intact2(2, :), DisplayName='CW, LC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Intact2(3, :), DisplayName='CCW, HC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Intact2(4, :), DisplayName='CCW, LC', LineWidth=1.5)
hold off
xlabel("orientation")
ylabel("choice count")
legend

subplot(2, 3, 2)
hold on
plot(uniqStim, nChoicesV1Damaged2(1, :), DisplayName='CW, HC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Damaged2(2, :), DisplayName='CW, LC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Damaged2(3, :), DisplayName='CCW, HC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Damaged2(4, :), DisplayName='CCW, LC', LineWidth=1.5)
hold off
xlabel("orientation")
ylabel("choice count")
legend

subplot(2, 3, 3)
hold on
plot(uniqStim, nChoicesV1Adjusted2(1, :), DisplayName='CW, HC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Adjusted2(2, :), DisplayName='CW, LC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Adjusted2(3, :), DisplayName='CCW, HC', LineWidth=1.5)
plot(uniqStim, nChoicesV1Adjusted2(4, :), DisplayName='CCW, LC', LineWidth=1.5)
hold off
xlabel("orientation")
ylabel("choice count")
legend


figure

subplot(2, 3, 1)
hold on
plot(rad2deg(uniqStim), psychometricFns, DisplayName="Intact V1", LineWidth=1.5)
plot(rad2deg(uniqStim), psychometricFnsDamaged, DisplayName="Damaged V1", LineWidth=1.5)
plot(rad2deg(uniqStim), psychometricFnsAdjusted, DisplayName="Adjusted V1", LineWidth=1.5)
hold off
xlabel("Orientation")
ylabel("prop CCW")
legend

subplot(2, 3, 2)
hold on
plot(rad2deg(uniqStim), confFns, DisplayName="Intact V1", LineWidth=1.5)
plot(rad2deg(uniqStim), confFnsV1Damaged, DisplayName="Damaged V1", LineWidth=1.5)
plot(rad2deg(uniqStim), confFnsV1Adjusted, DisplayName="Adjusted V1", LineWidth=1.5)
hold off
xlabel("Orientation")
ylabel("prop HC")
title("Bayesian")
legend

subplot(2, 3, 3)
hold on
plot(rad2deg(uniqStim), confFns2, DisplayName="Intact V1", LineWidth=1.5)
plot(rad2deg(uniqStim), confFnsV1Damaged2, DisplayName="Damaged V1", LineWidth=1.5)
plot(rad2deg(uniqStim), confFnsV1Adjusted2, DisplayName="Adjusted V1", LineWidth=1.5)
hold off
xlabel("Orientation")
ylabel("prop HC")
title("SDT")
legend

subplot(2, 3, 4)
hold on
plot(rad2deg(uniqStim), psychometricFnsByConf2(1, :), DisplayName="HC", LineWidth=1.5)
plot(rad2deg(uniqStim), psychometricFnsByConf2(2, :), DisplayName="LC", LineWidth=1.5)
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Intact V1")
legend

subplot(2, 3, 5)
hold on
plot(rad2deg(uniqStim), psychometricFnsDamagedByConf2(1, :), DisplayName="HC", LineWidth=1.5)
plot(rad2deg(uniqStim), psychometricFnsDamagedByConf2(2, :), DisplayName="LC", LineWidth=1.5)
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Damaged V1")
legend

subplot(2, 3, 6)
hold on
plot(rad2deg(uniqStim), psychometricFnsAdjustedByConf2(1, :), DisplayName="HC", LineWidth=1.5)
plot(rad2deg(uniqStim), psychometricFnsAdjustedByConf2(2, :), DisplayName="LC", LineWidth=1.5)
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Adjusted V1")
legend


%%

% Plot PDFs for a particular orientation to see if the decoded likelihood
% changes trial to trial for the same orientation
uniqStimVals = unique(stimVector);
fltTrlIdx    = (stimVector == deg2rad(100.5));
trlVals      = 1:ntrials;
fltTrialVals = trlVals(fltTrlIdx);

figure

subplot(4, 3, 1)
hold on

stdVals = zeros(numel(fltTrialVals), 1);
for j=1:numel(fltTrialVals)
    i = fltTrialVals(j);
    pdf = decodedPDFs{i}.pdf;
    x   = decodedPDFs{i}.x_deg;
    stdVals(j) = decodedPDFs{i}.sigma;

    plot(x, pdf)
end
xlabel("orientation")
ylabel("PDF")
title("Intact V1")
% xlim([60, 120])
hold off

subplot(4, 3, 4)
histogram(stdVals)
title("Std vals")

subplot(4, 3, 7)
histogram(trlConfVars(fltTrlIdx))
title("Conf var")

subplot(4, 3, 10)
histogram(trlConfVars2(fltTrlIdx))
title("Conf var")

subplot(4, 3, 2)
hold on

stdVals = zeros(numel(fltTrialVals), 1);
for j=1:numel(fltTrialVals)
    i = fltTrialVals(j);
    pdf = decodedPDFsV1Damaged{i}.pdf;
    x   = decodedPDFsV1Damaged{i}.x_deg;
    stdVals(j) = decodedPDFsV1Damaged{i}.sigma;
    
    plot(x, pdf)
end
xlabel("orientation")
ylabel("PDF")
title("Damaged V1")
hold off

subplot(4, 3, 5)
histogram(stdVals)
title("Std vals")

subplot(4, 3, 8)
histogram(trlConfVarsV1Damaged(fltTrlIdx))
title("Conf var")

subplot(4, 3, 11)
histogram(trlConfVarsV1Damaged2(fltTrlIdx))
title("Conf var")

subplot(4, 3, 3)
hold on

stdVals = zeros(numel(fltTrialVals), 1);
for j=1:numel(fltTrialVals)
    i = fltTrialVals(j);
    pdf = decodedPDFsV1Adjusted{i}.pdf;
    x   = decodedPDFsV1Adjusted{i}.x_deg;
    stdVals(j) = decodedPDFsV1Adjusted{i}.sigma;
    
    plot(x, pdf)
end
xlabel("orientation")
ylabel("PDF")
title("Adjusted V1")
hold off

subplot(4, 3, 6)
histogram(stdVals)
title("Std vals")

subplot(4, 3, 9)
histogram(trlConfVarsV1Ajusted(fltTrlIdx))
title("Conf var")

subplot(4, 3, 12)
histogram(trlConfVarsV1Ajusted2(fltTrlIdx))
title("Conf var")

% Fix PDF first
% Quantify confidence - Probability of being correct - choose decision -
% calculate probability - choose a confidence criteria 
% Other option to quantify confidence - get standard deviation of the PDF -
% compute confidence variable from the std val - use criteria to compute
% confidence. Things to check: does std changes from trial to trial or is it fixed (Look at the distribution)?

%% Perform optimization
stimVals            = rad2deg(uniqStimVals);
optParamsV1Intact   = optimizeCASANDRE(stimVals, nChoicesV1Intact2);
optParamsV1Damaged  = optimizeCASANDRE(stimVals, nChoicesV1Damaged2);
optParamsV1Adjusted = optimizeCASANDRE(stimVals, nChoicesV1Adjusted2);

% optParamsV1Intact =
% 
%     5.9120    0.3117    0.1694

% optParamsV1Damaged =
% 
%     9.0982    0.3031   10.5314

% optParamsV1Adjusted =
% 
%     6.2293    0.6561    3.8225

%%
modelParams.sigma_d = optParamsV1Intact(1);
modelParams.Cd      = 90;
modelParams.Cc      = optParamsV1Intact(2);
modelParams.sigma_m = optParamsV1Intact(3);

choicePDFsV1Intact  = getLLhChoice_CASANDRE(stimVals, modelParams);

modelParams.sigma_d = optParamsV1Damaged(1);
modelParams.Cd      = 90;
modelParams.Cc      = optParamsV1Damaged(2);
modelParams.sigma_m = optParamsV1Damaged(3);

choicePDFsV1Damaged  = getLLhChoice_CASANDRE(stimVals, modelParams);

modelParams.sigma_d = optParamsV1Adjusted(1);
modelParams.Cd      = 90;
modelParams.Cc      = optParamsV1Adjusted(2);
modelParams.sigma_m = optParamsV1Adjusted(3);

choicePDFsV1Adjusted  = getLLhChoice_CASANDRE(stimVals, modelParams);

% Choice PDFs
figure

subplot(2, 3, 1)
hold on

plot(stimVals, choicePDFsV1Intact(1, :), DisplayName='CW, HC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Intact(2, :), DisplayName='CW, LC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Intact(3, :), DisplayName='CCW, HC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Intact(4, :), DisplayName='CCW, LC', LineWidth=1.5)

hold off
xlabel("orientation")
ylabel("choice count")
title("intact V1")
legend

subplot(2, 3, 2)
hold on

plot(stimVals, choicePDFsV1Damaged(1, :), DisplayName='CW, HC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Damaged(2, :), DisplayName='CW, LC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Damaged(3, :), DisplayName='CCW, HC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Damaged(4, :), DisplayName='CCW, LC', LineWidth=1.5)

hold off
xlabel("orientation")
ylabel("choice count")
title("Damaged V1")
legend

subplot(2, 3, 3)
hold on

plot(stimVals, choicePDFsV1Adjusted(1, :), DisplayName='CW, HC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Adjusted(2, :), DisplayName='CW, LC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Adjusted(3, :), DisplayName='CCW, HC', LineWidth=1.5)
plot(stimVals, choicePDFsV1Adjusted(4, :), DisplayName='CCW, LC', LineWidth=1.5)

hold off
xlabel("orientation")
ylabel("choice count")
title("Adjusted V1")
legend


% Psychometric function
figure

subplot(3, 3, 1)
hold on
scatter(stimVals, psychometricFns)
psyFn = choicePDFsV1Intact(3,:) + choicePDFsV1Intact(4,:);
plot(stimVals, psyFn, LineWidth=2)
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Intact V1")
ylim([0 1])
legend

subplot(3, 3, 2)
hold on
scatter(stimVals, psychometricFnsDamaged)
psyFn = choicePDFsV1Damaged(3,:) + choicePDFsV1Damaged(4,:);
plot(stimVals, psyFn, LineWidth=2)
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Damaged V1")
ylim([0 1])
legend

subplot(3, 3, 3)
hold on
scatter(stimVals, psychometricFnsAdjusted)
psyFn = choicePDFsV1Adjusted(3,:) + choicePDFsV1Adjusted(4,:);
plot(stimVals, psyFn, LineWidth=2)
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Adjusted V1")
ylim([0 1])
legend


subplot(3, 3, 4)
hold on
scatter(stimVals, psychometricFnsByConf2(1, :))
scatter(stimVals, psychometricFnsByConf2(2, :))

psyFnHC = choicePDFsV1Intact(3,:) ./ (choicePDFsV1Intact(1,:) + choicePDFsV1Intact(3,:));
plot(stimVals, psyFnHC, LineWidth=2, DisplayName="HC")
psyFnLC = choicePDFsV1Intact(4,:) ./ (choicePDFsV1Intact(2,:) + choicePDFsV1Intact(4,:));
plot(stimVals, psyFnLC, LineWidth=2, DisplayName="LC")
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Intact V1")
legend


subplot(3, 3, 5)
hold on
scatter(stimVals, psychometricFnsDamagedByConf2(1, :))
scatter(stimVals, psychometricFnsDamagedByConf2(2, :))

psyFnHC = choicePDFsV1Damaged(3,:) ./ (choicePDFsV1Damaged(1,:) + choicePDFsV1Damaged(3,:));
plot(stimVals, psyFnHC, LineWidth=2, DisplayName="HC")
psyFnLC = choicePDFsV1Damaged(4,:) ./ (choicePDFsV1Damaged(2,:) + choicePDFsV1Damaged(4,:));
plot(stimVals, psyFnLC, LineWidth=2, DisplayName="LC")
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Damaged V1")
legend

subplot(3, 3, 6)
hold on
scatter(stimVals, psychometricFnsAdjustedByConf2(1, :))
scatter(stimVals, psychometricFnsAdjustedByConf2(2, :))

psyFnHC = choicePDFsV1Adjusted(3,:) ./ (choicePDFsV1Adjusted(1,:) + choicePDFsV1Adjusted(3,:));
plot(stimVals, psyFnHC, LineWidth=2, DisplayName="HC")
psyFnLC = choicePDFsV1Adjusted(4,:) ./ (choicePDFsV1Adjusted(2,:) + choicePDFsV1Adjusted(4,:));
plot(stimVals, psyFnLC, LineWidth=2, DisplayName="LC")
hold off
xlabel("Orientation")
ylabel("prop CCW")
title("Adjusted V1")
legend

% Confidence function
subplot(3, 3, 7)
hold on
scatter(stimVals, confFns2)

prop_HC = choicePDFsV1Intact(1, :) + choicePDFsV1Intact(3, :);
plot(stimVals, prop_HC, LineWidth=2)
hold off
xlabel("Orientation")
ylabel("%HC")
title("Intact V1")
legend

subplot(3, 3, 8)
hold on
scatter(stimVals, confFnsV1Damaged2)

prop_HC = choicePDFsV1Damaged(1, :) + choicePDFsV1Damaged(3, :);
plot(stimVals, prop_HC, LineWidth=2)
hold off
xlabel("Orientation")
ylabel("%HC")
title("Damaged V1")
legend

subplot(3, 3, 9)
hold on
scatter(stimVals, confFnsV1Adjusted2)

prop_HC = choicePDFsV1Adjusted(1, :) + choicePDFsV1Adjusted(3, :);
plot(stimVals, prop_HC, LineWidth=2)
hold off
xlabel("Orientation")
ylabel("%HC")
title("Adjusted V1")
legend

%% Utility functions

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


