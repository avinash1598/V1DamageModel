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

trialDecisionsDamaged = zeros(1, ntrials);
neuronSpikeResponsesDamaged = false(ntrials, nNeurons, length(timeBins)); % Creating a logical matrix to save memory

trialDecisionsDamaged_Adjusted = zeros(1, ntrials);
neuronSpikeResponsesDamaged_Adjusted = false(ntrials, nNeurons - nrnCntDamaged, length(timeBins)); % Creating a logical matrix to save memory

% Tuning params just for intact neurons
tuningParamsAdjusted.d = tuningParams.d(intactNrnIdxes);                      % Direction selectivity (fixed at 0)
tuningParamsAdjusted.alpha = tuningParams.alpha(intactNrnIdxes);              % Aspect ratio (fixed at 2)
tuningParamsAdjusted.b = tuningParams.b(intactNrnIdxes);                      % Controls sharpness of tuning curve (fixed at 2)
tuningParamsAdjusted.q = tuningParams.q(intactNrnIdxes);                      % Controls amplitude of peak firing rate (variable)
tuningParamsAdjusted.w = tuningParams.w(intactNrnIdxes);                      % Unused parameter (set to 1)
tuningParamsAdjusted.UNTUNED_FILTER_AMPL = tuningParams.UNTUNED_FILTER_AMPL;  % Untuned filter amplitude (fixed at 0)
tuningParamsAdjusted.eps1 = tuningParams.eps1(intactNrnIdxes);                % Controls dynamic range (variable)
tuningParamsAdjusted.beta = tuningParams.beta(intactNrnIdxes);                % Controls dynamic range (variable)

% Plot orientation tuning of damaged and intact neurons
stimVector_tuningFn = linspace(0, pi, 200);
tuningFns = orientationTunedFiringRate(stimVector_tuningFn, ...
    neuronsPrefOrientation, tuningParams);
tuningFns = tuningFns';

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

    % Spontaneous noise different for each trial
    spontaneousNoise = lognrnd(1, 0.8, nNeurons, length(timeBins)); %length(timeBins) % This is already included in the model! Maybe this is not needed for intact neurons. Only for damaged neurons.
    
    % Modulate gain of current trial based on previous trial
    % nNeurons x No time bins
    trlGainVector = squeeze(repmat(gainVector(trialIDx, :), [1, 1, length(timeBins)])); % Extract gain vector for this trial for each timebin
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Intact
    % STEP 2: Compute stimulus response for each neuron over time
    % This multiplies the firing rates with a time-dependent stimulus response profile
    % Output: 
    %  - trlStimResponse: response of each neuron over time for each trial (nNeurons x nTimeBins)
    trlStimResponse = firingRates(trialIDx, :)'.*stimRespProfile;
    
    % Add background spontaneous noise
    trlStimResponse = trlStimResponse + spontaneousNoise; % Do i need to add this for this simulation?
    
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
    
    % Decision: CW (-1) or CCW (1) based on decoded orientation
    decision = (thetaMLE > pi/2)*(-1) + (thetaMLE <= pi/2)*(1);
    
    % Store spike trains and decision results
    neuronSpikeResponses(trialIDx, :, :) = logical(spikes);  % Store spikes
    trialDecisions(trialIDx) = decision;  % Store decision result (CW or CCW)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V1 damaged
    % STEP 2: Compute stimulus response for each neuron over time
    % This multiplies the firing rates with a time-dependent stimulus response profile
    % Output: 
    %  - trlStimResponse: response of each neuron over time for each trial (nNeurons x nTimeBins)
    trlStimResponse = firingRates(trialIDx, :)'.*stimRespProfile;
    
    % Simulate V1 damage
    trlStimResponse(damagedNrnIdxes, :) = 0;
    
    % Add background spontaneous noise
    trlStimResponse = trlStimResponse + spontaneousNoise; % Do i need to add this for this simulation?
    
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
    thetaMLE = decodeOrientationFromSpikes(spikes, ...
        neuronsPrefOrientation, params, tuningParams);
    % decodingError = thetaMLE - noisyStimVector(trialIDx);
    
    % Decision: CW (-1) or CCW (1) based on decoded orientation
    decision = (thetaMLE(end) > pi/2)*(-1) + (thetaMLE(end) <= pi/2)*(1);
    
    % Store spike trains and decision results
    neuronSpikeResponsesDamaged(trialIDx, :, :) = logical(spikes);  % Store spikes
    trialDecisionsDamaged(trialIDx) = decision;  % Store decision result (CW or CCW)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Adjusted decoder
    % STEP 3: Generate modulated Poisson spikes for each trial
    % Output:
    %  - spikes: spike trains for each neuron in the trial
    %  - modStimResponse: modified stimulus response after gain modulation
    params = struct();
    params.timeStep = timeStep;
    params.timeBins = timeBins;
    params.nNeurons = nNeurons - nrnCntDamaged;
    
    % STEP 4: Decode the stimulus orientation based on the spike trains
    % Output:
    %  - thetaMLE: maximum likelihood estimate of stimulus orientation based on spikes
    %  - decodingError: error between decoded orientation and actual stimulus
    spikesAdjusted = spikes(intactNrnIdxes, :);
    neuronsPrefOrientationAdjusted = neuronsPrefOrientation(intactNrnIdxes);
    params.stimDuration = stimDuration;
    thetaMLE = decodeOrientationFromSpikes(spikesAdjusted, ...
        neuronsPrefOrientationAdjusted, params, tuningParamsAdjusted);
    % decodingError = thetaMLE - noisyStimVector(trialIDx);
    
    % Decision: CW (-1) or CCW (1) based on decoded orientation
    decision = (thetaMLE(end) > pi/2)*(-1) + (thetaMLE(end) <= pi/2)*(1);
    
    % Store spike trains and decision results
    neuronSpikeResponsesDamaged_Adjusted(trialIDx, :, :) = logical(spikesAdjusted);  % Store spikes
    trialDecisionsDamaged_Adjusted(trialIDx) = decision;  % Store decision result (CW or CCW)

end

% ----------------------------------
% Plot results
% ----------------------------------

figure
subplot(2, 3, 1)
hold on
for nIDx = intactNrnIdxes'
    plot(rad2deg(stimVector_tuningFn), tuningFns(nIDx, :));
end
xlabel("Ori (deg)")
ylabel("IPS")
hold off
title("Tuning function for intact neurons")

subplot(2, 3, 2)
hold on
for nIDx = damagedNrnIdxes'
    plot(rad2deg(stimVector_tuningFn), tuningFns(nIDx, :));
end
xlabel("Ori (deg)")
ylabel("IPS")
hold off
title("Tuning function for damaged neurons")

subplot(2, 3, 3)
hold on
title("Single trial spikes" + newline + "all neurons")
imagesc(squeeze(spikes(:, :))), colormap(flipud('gray'))
axis square
box off, axis off
xlabel("Time (s)")
ylabel("Spikes")
hold off

uniqStim = unique(stimVector);
psychometricData = zeros(1, length(uniqStim));
psychometricDataDamaged = zeros(1, length(uniqStim));
psychometricDataDamaged_Adjusted = zeros(1, length(uniqStim));

for i=1:stimParam.numStim
    stimOrientation = uniqStim(i);
    givenOrientationTrialIDxes = find(stimVector == stimOrientation);

    % Intact V1
    decision = trialDecisions(givenOrientationTrialIDxes);
    
    % Psychometric data
    percent_CW = length(find(decision == -1)) / length(decision);
    psychometricData(i) = percent_CW;
    
    % Damaged V1
    decision = trialDecisionsDamaged(givenOrientationTrialIDxes);

    % Psychometric data
    percent_CW = length(find(decision == -1)) / length(decision);
    psychometricDataDamaged(i) = percent_CW;

    % Damaged and adjusted V1
    decision = trialDecisionsDamaged_Adjusted(givenOrientationTrialIDxes);

    % Psychometric data
    percent_CW = length(find(decision == -1)) / length(decision);
    psychometricDataDamaged_Adjusted(i) = percent_CW;

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

x = rad2deg(uniqStim);
y = psychometricDataDamaged;

% Psychometric function
psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
b0 = [90, std(x)]; % Initial guess

lb = [-Inf, eps];
ub = [ Inf, Inf];

opts = optimoptions('lsqcurvefit','Display','off');
bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);

xFit = linspace(min(x), max(x), 200);
yFit = psyFun(bHat, xFit);

hold on
scatter(x, y, 'DisplayName', 'Data')
plot(xFit, yFit, 'k', 'LineWidth', 2, LineStyle='--')
xlabel("Orientation (deg)")
ylabel("% CCW")
title('Psychometric Function');
hold off


x = rad2deg(uniqStim);
y = psychometricDataDamaged_Adjusted;

% Psychometric function
psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
b0 = [90, std(x)]; % Initial guess

lb = [-Inf, eps];
ub = [ Inf, Inf];

opts = optimoptions('lsqcurvefit','Display','off');
bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);

xFit = linspace(min(x), max(x), 200);
yFit = psyFun(bHat, xFit);

hold on
scatter(x, y, 'DisplayName', 'Data')
plot(xFit, yFit, 'k', 'LineWidth', 2, LineStyle='-.')
xlabel("Orientation (deg)")
ylabel("% CCW")
title('Psychometric Function');
hold off


