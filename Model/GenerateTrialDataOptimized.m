%% 
clear all
close all
% clc

rng('shuffle'); % Ensures that different random numbers are generated every time

% TIP: Keep all possible 1D variables as row vectors.
% NOTE: Time step should be small relative to the spike rate of a single neuron,
% so that the probability of firing doesn't shoot up. 

% TODO: Add a check to ensure that the firing rate is not too large 
% compared to the time window.

% ----------------------------------
% Parameters
% ----------------------------------
nNeurons = 30;        % Number of neurons
stimDuration = 1;      % Stimulus duration in seconds
varGain = 0.5;         % Variance in gain for modulated Poisson process
timeStep = 0.001;      % Time step (1ms) for binning the stimulus duration

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
% TODO: Make this flat instead
% stimRespProfile = [0, normcdf(timeBins(2:end), .25, .05) .* 1./timeBins(2:end)];
stimRespProfile = 1 + zeros(1, numel(timeBins));

% Gain vector for modulated Poisson process (randomized for each trial, neuron)
% Time bin based modulation is not included in this initialization to avoid memory problems.
% gainVector = gamrnd(1./varGain, varGain, [ntrials, nNeurons]);
gainVector = 1 + zeros(ntrials, nNeurons); % constant gain - NO gain modulation

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
tuningParams.beta(:) = lognrnd(2.5, 0.5, 1, nNeurons);   % Random values for beta parameter
tuningParams.eps1(:) = lognrnd(1, 0.8, 1, nNeurons);     % Random values for dynamic range control

% Assign random preferred orientations to the neurons
neuronsPrefOrientation(:) = pi * rand(1, nNeurons);  % Random orientations from -pi to pi

% Structures to store final neuron spikes
% Preallocate result and spike response matrices
trialDecisions = zeros(1, ntrials);
neuronSpikeResponses = false(ntrials, nNeurons, length(timeBins)); % Creating a logical matrix to save memory

% ----------------------------------
% Computing stimulus response begins
% ----------------------------------

% STEP 1: Compute orientation-tuned firing rates for each trial
% Output: 
%  - firingRates: matrix of firing rates (nTrials x nNeurons)
firingRates = orientationTunedFiringRate(noisyStimVector, ...
    neuronsPrefOrientation, tuningParams);


for trialIDx = 1:ntrials
    if mod(trialIDx, stimParam.countPerStim) == 0
        disp(trialIDx)
    end

    % STEP 2: Compute stimulus response for each neuron over time
    % This multiplies the firing rates with a time-dependent stimulus response profile
    % Output: 
    %  - trlStimResponse: response of each neuron over time for each trial (nNeurons x nTimeBins)
    trlStimResponse = firingRates(trialIDx, :)'.*stimRespProfile;
    
    % Add background spontaneous noise
    trlStimResponse = trlStimResponse + lognrnd(1, 0.8, nNeurons, length(timeBins)); % Do i need to add this for this simulation?
    
    % Modulate gain of current trial based on previous trial
    % nNeurons x No time bins
    trlGainVector = squeeze(repmat(gainVector(trialIDx, :), [1, 1, length(timeBins)])); % Extract gain vector for this trial for each timebin
    
    % STEP 3: Generate modulated Poisson spikes for each trial
    % Output:
    %  - spikes: spike trains for each neuron in the trial
    %  - modStimResponse: modified stimulus response after gain modulation
    params = struct();
    params.timeStep = timeStep;
    params.timeBins = timeBins;
    params.nNeurons = nNeurons;
    [spikes, modStimResponse] = generateModulatedPoissonSpikes(trlStimResponse, ...
        trlGainVector, params);
    
    % STEP 4: Decode the stimulus orientation based on the spike trains
    % Output:
    %  - thetaMLE: maximum likelihood estimate of stimulus orientation based on spikes
    %  - decodingError: error between decoded orientation and actual stimulus
    params.stimDuration = stimDuration;
    thetaMLE = decodeOrientationFromSpikes(spikes, ...
        neuronsPrefOrientation, params, tuningParams);
    decodingError = thetaMLE - noisyStimVector(trialIDx);
    
    % Decision: CW (-1) or CCW (1) based on decoded orientation
    decision = (thetaMLE(end) > pi/2)*(-1) + (thetaMLE(end) <= pi/2)*(1);
    
    % Store spike trains and decision results
    neuronSpikeResponses(trialIDx, :, :) = logical(spikes);  % Store spikes
    trialDecisions(trialIDx) = decision;  % Store decision result (CW or CCW)
end

% % ----------------------------------
% % Store results/responses
% % ----------------------------------
% trialMatrix = zeros(ntrials, 4); % trial no, stim orientation, estimated decision
% 
% for trial_idx = 1:ntrials
%     trialMatrix(trial_idx, 1) = trial_idx;                               % Trial index
%     trialMatrix(trial_idx, 2) = rad2deg(stimVector(trial_idx));          % Stimulus orientation
%     trialMatrix(trial_idx, 3) = rad2deg(noisyStimVector(trial_idx));     % Noisy stimulus orientation (actual)
%     trialMatrix(trial_idx, 4) = trialDecisions(trial_idx);               % Decision 
% end
% 
% expData.trialMatrix = trialMatrix;
% expData.columnDescriptions.trialMatrix = {'Column 1: Trial Number', 'Column 2: Stimulus orientation (in degree)', 'Column 3: Noisy stimulus orientation (in degree)', 'Column 4: Decision (1: CW, -1: CCW)'};
% expData.preferredOrientation = rad2deg(neuronsPrefOrientation);
% expData.trialResponses = neuronSpikeResponses;
% expData.columnDescriptions.trialResponses = {'Spike responses for ntrials x nNeurons x nTimeBins. The values are stored in a logical matrix. Since each element in the matrix can be either zero or one, the logical matrix uses one bit of memory per element, instead of one byte used by standard matrix.'};
% expData.timeBins = timeBins;
% 
% save('./Data/expData.mat', 'expData', '-v7.3');

% ----------------------------------
% Plot results
% ----------------------------------

figure
subplot(2, 3, 1)
hold on
title("Single trial response" + newline + "all neurons")
for nIDx = 1:nNeurons
    plot(squeeze(timeBins), squeeze(modStimResponse(nIDx, :)));
end
axis square
xlabel("Time (s)")
ylabel("IPS")
hold off

subplot(2, 3, 2)
hold on
title("Single trial spikes" + newline + "all neurons")
imagesc(squeeze(spikes(:, :))), colormap(flipud('gray'))
axis square
box off, axis off
xlabel("Time (s)")
ylabel("Spikes")
hold off

subplot(2,3,3)
hold on
title("Single trial")
timePts = linspace(0, stimDuration, length(decodingError));
plot(timePts, decodingError)
xlabel("Time (s)")
ylabel("Decoding error")
hold off

uniqStim = unique(stimVector);
psychometricData = zeros(1, length(uniqStim));
percentCorrect   = zeros(1, length(uniqStim));

for i=1:stimParam.numStim
    stimOrientation = uniqStim(i);
    givenOrientationTrialIDxes = find(stimVector == stimOrientation);
    decision = trialDecisions(givenOrientationTrialIDxes);
    
    actualDecision = zeros(size(decision));
    actualDecision( noisyStimVector(givenOrientationTrialIDxes) < deg2rad(90) ) =  1;  % CCW
    actualDecision( noisyStimVector(givenOrientationTrialIDxes) > deg2rad(90) ) = -1; % CW
    
    % Psychometric data
    percent_CW = length(find(decision == -1)) / length(decision);
    psychometricData(i) = percent_CW;
    
    % Percent correct
    percentCorrect(i) = sum(actualDecision == decision) / numel(decision);
end

x = rad2deg(uniqStim);
y = psychometricData;

% Psychometric function
psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
b0 = [90, std(x)]; % Initial guess

lb = [-Inf, eps];
ub = [ Inf, Inf];

opts = optimoptions('lsqcurvefit','Display','off');
bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);

xFit = linspace(min(x), max(x), 200);
yFit = psyFun(bHat, xFit);

subplot(2,3,4)
scatter(x, y, 'DisplayName', 'Data')
hold on
plot(xFit, yFit, 'k', 'LineWidth', 2)
hold off
xlabel("Orientation (deg)")
ylabel("% CCW")
title('Psychometric Function');
hold off

% Detection threshold
% percent correct 
x = ( rad2deg(uniqStim) - 90 );
y = percentCorrect;

subplot(2,3,5)
scatter(x, y, 'DisplayName', 'Data')
% hold on
% plot(xFit, yFit, 'k', 'LineWidth', 2)
% hold off
xlabel("Orientation (deg)")
ylabel("% correct")
title('Psychometric Function');
hold off


% %%
% % Sanity check: Mean vs variance plot
% numIntervals = 50;
% intervalSize = floor(length(timeBins) / numIntervals);  % 20 columns per interval
% meanSpkCnt = zeros(nNeurons, numIntervals);
% varSpkCnt  = zeros(nNeurons, numIntervals);
% 
% for nIdx = 1:nNeurons
%     spkForThisNrn = squeeze( neuronSpikeResponses(:, nIdx, :) );
%     
%     for i = 1:numIntervals
%         startCol = (i-1) * intervalSize + 1;
%         endCol = min(i * intervalSize, length(timeBins));  % Handle the last interval
%     
%         % mean spk rate
%         spkCntInThisInterval = sum( spkForThisNrn(:, startCol:endCol), 2 );
%         intervalData = mean( spkCntInThisInterval ); 
%         
%         meanSpkCnt(nIdx, i) = mean( spkCntInThisInterval );
%         varSpkCnt(nIdx, i)  = var( spkCntInThisInterval );
%     end
% 
% end
% 
% figure
% scatter(meanSpkCnt(:), varSpkCnt(:))
% hold on
% plot([0 1], [0 1], 'k--') 
% hold off
% xlabel("Mean spk cnt")
% ylabel("Var spk cnt")
