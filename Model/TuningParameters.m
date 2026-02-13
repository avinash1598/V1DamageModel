clear all
close all
clc

rng('shuffle');

% TIP: Keep all possible 1D variables to row vector
% Note: Time step should be set carefully. It should be pretty small
% relative to the spike rate of single neuron so that the probability of
% firing does not shoot up.
% TODO: add a check to make sure firing rate is not so big compared to the
% time window.

% ----------------------------------
% Params
% ----------------------------------
nNeurons = 100;      % Count of neurons
stimDuration = 1;    % Stimulus duration set to 1 seconds
varGain = 0.5;       % Variance in gain for modulated poisson process
timeStep = 0.001;    % 0.001s (1ms) - Step size of time bins used for binning stimulus duration 
% ntrials = 100;     % Offloaded to generateStim function. No of stimulus to simulate the model with

% stimParam.startInterval = pi/2-pi/4;
% stimParam.endInterval = pi/2+pi/4;
stimParam.startInterval = deg2rad(90-5);
stimParam.endInterval = deg2rad(90+5);
stimParam.numStim = 20;
stimParam.countPerStim = 2;
ntrials = stimParam.numStim*stimParam.countPerStim;

neuronsPrefOrientation = zeros(1, nNeurons);

% stimVector = generateStimVector(stimParam); % Stimulus vector containing orientation for each trial
stimNoise = 0 + 0.1 * randn(1, ntrials); % What is std dev here? 5.73 degrees
stimVector = repelem(linspace(stimParam.startInterval, stimParam.endInterval, ...
    stimParam.numStim), stimParam.countPerStim);  % Vector of stimuli
noisyStimVector = stimVector + stimNoise;         % Noisy stimulus vector

% Shuffle stim vector - this is important for studying effect of top down
% effect
shuffleIdx = randperm(ntrials);
% stimVector = stimVector(shuffleIdx);
noisyStimVector = noisyStimVector(shuffleIdx); 
stimVector = noisyStimVector;

timeBins = 0:timeStep:stimDuration; 
% stimRespProfile = [0, normcdf(timeBins(2:end), .25, .05) .* 1./timeBins(2:end)];
stimRespProfile = 1 + zeros(1, numel(timeBins));
gainVector = repmat(gamrnd(1./varGain, varGain, [ntrials, nNeurons]), [1, 1, length(timeBins)]); % for modulated poisson process - gain for each trial, neuron, time bin
% gainVector = 1 + zeros(ntrials, nNeurons); % constant gain - NO gain modulation
gainVector(:) = 1; % No gain modulation

% Neurons tuning parameters
tuningParams.d = zeros(1, nNeurons) + 1;       % (fixed) Direction selectivity - set it to zero (no need for neuron to be directional selective).
tuningParams.alpha = zeros(1, nNeurons) + 2;   % (fixed) Aspect ratio - controls sharpness. Keep this fixed. Reducing the value makes the changes very rapid towards the end which we probably don't want.
tuningParams.b = zeros(1, nNeurons) + 2;       % (fixed maybe/variable - (0.5, some max - 3, 4 ...)) Control this - Control sharpness + range of the neuron. Set it to 2 for these simulations
tuningParams.q = zeros(1, nNeurons) + 1;       % (variable) Set it to some constant. Controls the sharpness and amplitude of peak FR.
tuningParams.w = zeros(1, nNeurons) + 1;       % (fixed) Doesn't matter what is val is becz untuned filter amp is zero.
tuningParams.UNTUNED_FILTER_AMPL = 0;          % (fixed) Untuned filter not needed.
tuningParams.eps1 = zeros(1, nNeurons);        % (variable) Controls dynamic range.
tuningParams.beta = zeros(1, nNeurons) + 1;    % (variable) Controls dynamic range.

% Overriding variable tuning parameters
% tuningParams.b(:) = 0.5 + 10*rand(1, nNeurons);        % Maybe change non linearity instead. This is derivative order so ideally should be integer. Range: 0.5-4
% tuningParams.q(:) = evrnd(5.5, 1.5, [1, nNeurons]);    % 8*randn(1, nNeurons);                 % Range: 0 - 20 ips, gamrnd(2, 2, [1, nNeurons]);
tuningParams.q(:) = 8 * rand(1, nNeurons);               % 8*randn(1, nNeurons);                 % Range: 0 - 20 ips, gamrnd(2, 2, [1, nNeurons]);
% tuningParams.beta(:) = lognrnd(2.5, 0.5, 1, nNeurons);   % lognrnd(mu, sigma, 1, nneurons)
tuningParams.beta(:) = lognrnd(2.5, 0.5, 1, nNeurons);   % lognrnd(mu, sigma, 1, nneurons)
tuningParams.eps1(:) = lognrnd(1, 0.8, 1, nNeurons);     % Range: 0 - 10 ips, Might not be correct initilization

% Temporary code
neuronsPrefOrientation(:) = pi * rand(1, nNeurons);   % Randomly choose neurons preferred orientation from -pi to pi
% neuronsPrefOrientation(:) = -pi + (2 * pi) * rand(1, nNeurons);   % Randomly choose neurons preferred orientation from -pi to pi

% ----------------------------------
% Computing stimulus response begins
% ----------------------------------

% STEP1: 
firingRates = orientationTunedFiringRate(stimVector, ...
    neuronsPrefOrientation, tuningParams);

% STEP2: 
% Find the time response of all these neurons to input stimuli
% This firing rate matrix will be multipled with some time response
% function to get the final stimulus response of each neuron.
% Perform elementwise multipliction between stimulus dependent firing rates
% and stimulus response profile.
stimResponse = firingRates.*reshape(stimRespProfile, 1, 1, length(stimRespProfile));

resultMat = zeros(stimParam.numStim, stimParam.countPerStim);
stimVals = zeros(1, stimParam.numStim);

for i = 1:stimParam.numStim
    stimVal_ = 0;
    for j=1:stimParam.countPerStim
        % STEP3:
        % Trial by trial processing - for added complexity at later point
        % Drive outpout of modulated poisson process
        trialIDx = (i-1)*stimParam.countPerStim + j;
        trlStimResponse = squeeze(stimResponse(trialIDx, :, :));
        trlGainVector = squeeze(gainVector(trialIDx, :, :));
        
        params = struct();
        params.timeStep = timeStep;
        params.timeBins = timeBins;
        params.nNeurons = nNeurons;
        [spikes, modStimResponse] = generateModulatedPoissonSpikes(trlStimResponse, ...
            trlGainVector, params);
        
        % STEP4:
        % Decode orientation based on modulated spikes
        params = struct();
        params.timeStep = timeStep;
        params.timeBins = timeBins;
        params.nNeurons = nNeurons;
        params.stimDuration = stimDuration;
        
        thetaMLE = decodeOrientationFromSpikes(spikes, ...
            neuronsPrefOrientation, params, tuningParams);
        decodingError = thetaMLE - stimVector(trialIDx);
        
        decision = (thetaMLE(end) > pi/2)*(1) + (thetaMLE(end) <= pi/2)*(-1);
        % actualVal = (stimVector(trialIDx) > pi/2)*(1) + (stimVector(trialIDx) <= pi/2)*(-1);

        resultMat(i, j) = (decision == 1);
        stimVal_ = stimVal_ + stimVector(trialIDx);
    end
    stimVals(i) = stimVal_/stimParam.countPerStim;
end

% Orientation selectivity index for each neuron
thetas_ = linspace(0, pi, 180);
tuningFnFR_ = orientationTunedFiringRate(thetas_, neuronsPrefOrientation, tuningParams)';
OSI = abs(tuningFnFR_*(exp(1i*(2*thetas_))'))./sum(abs(tuningFnFR_), 2);
peakSpkRt = squeeze(max(tuningFnFR_, [], 2)');

figure
subplot(2, 2, 1)
hold on
histogram(OSI, 10, 'Normalization', 'probability')
xlabel('Orientation Selectivity')
ylabel('Probability')
xlim([0, 1])
hold off

% Plot all tuning curves
subplot(2, 2, 2)
hold on
title("Tuning curves" + newline + " (all neurons)")
for i = 1:nNeurons
    plot(rad2deg(thetas_), tuningFnFR_)
end
xlabel('Orientation (deg)')
ylabel('ips')
xlim([0, 180])
hold off


subplot(2, 2, 3)
hold on
histogram(peakSpkRt, 10, 'Normalization', 'probability')
xlabel('Peak IPS (evoked)')
ylabel('Probability')
hold off

subplot(2, 2, 4)
hold on
histogram(tuningParams.eps1, 10, 'Normalization', 'probability')
xlabel('Baseline FR (spontaneous)')
ylabel('Probability')
hold off
