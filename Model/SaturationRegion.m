%% 
clear all
close all
% clc

rng('shuffle'); % Ensures that different random numbers are generated every time

% nrnCnts    = [5 10 15 20 25 30 40 50 75 100 150 200 250 300 400 500];
nrnCnts    = [5 10 15 20 25 30 40 50 75 100 500];
thresholds = zeros(numel(nrnCnts), 50);
psychFns   = zeros(numel(nrnCnts), 50, 200);

% Loop through each population size
for idx = 1:numel(nrnCnts)

fprintf("Population size %d", nrnCnts(idx));

for itr = 1:50
    
    disp(itr);

    nrnCnt = nrnCnts(idx);
    
    % ----------------------------------
    % Parameters
    % ----------------------------------
    nNeurons = nrnCnt;     % Number of neurons
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
        % if mod(trialIDx, stimParam.countPerStim) == 0
        %     disp(trialIDx)
        % end
        
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
    
    thresholds(idx, itr) = bHat(2);
    psychFns(idx, itr, :) = yFit;

end

end

%%
% Plot result
figure
subplot(2, 2, 1)
plot(nrnCnts, thresholds)
xlabel("Neuron count")
ylabel("Std of CDF fit")
% ylim([0 6])

subplot(2, 2, 2)
hold on
for i=1:3:numel(nrnCnts)
    plot(xFit, psychFns(i, :), DisplayName=""+nrnCnts(i), LineWidth=1.5);
end
xlabel("Neuron count")
ylabel("Std of CDF fit")
legend
hold off

%% Plot for individual neuron cnt
load("data.mat")
psychFns = data.psychFns;
idxes = [1 3 5 7 10];

% Plot psychometric curve for each iteration
figure
for i=1:numel(nrnCnts)
    subplot(3, 4, i)
    hold on

    for j=1:50
        plot(xFit, squeeze( psychFns(i, j, :)) )
    end

    hold off
    xlabel("Orientation (deg)")
    ylabel("% CCW")
    title(sprintf("Population size %d", nrnCnts(i)))

end

% Plot average psychometric function and plot threshold
thresholds = zeros(numel(nrnCnts), 50);

for i=1:numel(nrnCnts)
    for j = 1:50
        x = xFit;
        y = squeeze( psychFns(i, j, :) )';
        
        if i == 11 && j > 19
            continue
        end
    
        % Fit psychometric function
        psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
        b0 = [90, std(x)]; % Initial guess
        
        lb = [-Inf, eps];
        ub = [ Inf, Inf];
        
        opts = optimoptions('lsqcurvefit','Display','off');
        bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);
    
        thresholds(i, j) = bHat(2);
    end
end

figure

subplot(2, 2, 1)
hold on

for i=idxes
    x = xFit;
    y = mean( squeeze( psychFns(i, :, :) ), 1 );
    
    if i == 11
        y = mean( squeeze( psychFns(i, 1:19, :) ), 1 );
    end
    
    plot(x, y, LineWidth=1.5, DisplayName=""+nrnCnts(i))
end

hold off
xlabel("Orientation (deg)")
ylabel("Prop CCW")
title("Avg Psychometric function at various population size")
legend

subplot(2, 2, 3)
z = mean(thresholds, 2);
z(11) = sum(thresholds(11, :)) / 19;
plot(1:numel(nrnCnts), z, LineWidth=1.5)
xticks(1:numel(nrnCnts))
xticklabels(nrnCnts)
xlabel("Population size")
ylabel("Threshold (std dev of CDF fit)")

%%
data.psychFns = psychFns;
save('data.mat', 'data')
