%% 
% clear all
% close all
% clc

rng('shuffle'); % Ensures that different random numbers are generated every time

propDamaged_list         = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
% propDamaged_list         = [0.95, 0.98];
nItr                     = 50;
psychometricFns          = zeros(numel(propDamaged_list), nItr, 21);
psychometricFnsDamaged   = zeros(numel(propDamaged_list), nItr, 21);
psychometricFnsAdjusted  = zeros(numel(propDamaged_list), nItr, 21);

for didx = 7:numel(propDamaged_list)
    
    for itr = 1:nItr
        
        fprintf("Prop damaged :%.2f, Itr: %d", propDamaged_list(didx), itr);
        fprintf('\n')
        
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
        propDamaged = propDamaged_list(didx); %0.8;     % Proportion of damaged neurons
        
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
        trialDecisions = zeros(1, ntrials);
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
        
            [spikes, decision] = decodeDecision(gainVector, trialIDx, timeBins, firingRates, ...
                stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, tuningParams);
            
            % Store spike trains and decision results
            % neuronSpikeResponses(trialIDx, :, :) = logical(spikes);  % Store spikes
            trialDecisions(trialIDx) = decision;  % Store decision result (CW or CCW)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V1 damaged
            % Tuning function used for decoding should be same as
            % the previous one. The tuning function used for generating spikes
            % should be the damaged one.
            % Spikes are generated as per damaged neurons whereas decoding is done
            % as per original tuning function.
            % [spikes, decision] = decodeDecision(gainVector, trialIDx, timeBins, firingRatesV1Damaged, ...
            %     stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, tuningParamsV1Damaged);
            [spikes, decision] = decodeDecision(gainVector, trialIDx, timeBins, ...
                firingRatesV1Damaged, ... % Firing rates for damaged population
                stimRespProfile, timeStep, nNeurons, stimDuration, neuronsPrefOrientation, ...
                tuningParams); % Orignal tuning function
            
            % Store spike trains and decision results
            % neuronSpikeResponsesDamaged(trialIDx, :, :) = logical(spikes);  % Store spikes
            trialDecisionsDamaged(trialIDx) = decision;  % Store decision result (CW or CCW)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Adjusted decoder
            % Maybe use the same spike train as previous spikes obtained
            [spikes, decision] = decodeDecision( ...
                gainVector(:, intactNrnIdxes), ...
                trialIDx, timeBins, firingRatesAdjusted, ...
                stimRespProfile, timeStep, numel(intactNrnIdxes), stimDuration, ...
                neuronsPrefOrientation(intactNrnIdxes), tuningParamsAdjusted);
            
            % Store spike trains and decision results
            % neuronSpikeResponsesAdjusted(trialIDx, :, :) = logical(spikes);  % Store spikes
            trialDecisionsAdjusted(trialIDx) = decision;  % Store decision result (CW or CCW)

        end
        
        uniqStim = unique(stimVector);
        
        for i=1:stimParam.numStim
            stimOrientation = uniqStim(i);
            givenOrientationTrialIDxes = find(stimVector == stimOrientation);
            
            % Intact V1
            decision = trialDecisions(givenOrientationTrialIDxes);
            
            % Psychometric data
            percent_CW = length(find(decision == -1)) / length(decision);
            psychometricFns(didx, itr, i) = percent_CW; % This is actually CCW but not a big deal
            
            % Damaged V1
            decision = trialDecisionsDamaged(givenOrientationTrialIDxes);
        
            % Psychometric data
            percent_CW = length(find(decision == -1)) / length(decision);
            psychometricFnsDamaged(didx, itr, i) = percent_CW;
            
            % Damaged and adjusted V1
            decision = trialDecisionsAdjusted(givenOrientationTrialIDxes);
            
            % Psychometric data
            percent_CW = length(find(decision == -1)) / length(decision);
            psychometricFnsAdjusted(didx, itr, i) = percent_CW;
        
        end
        
%         % TODO: fit psychometric functions later
%         data.psychometricFns         = psychometricFns;
%         data.psychometricFnsDamaged  = psychometricFnsDamaged;
%         data.psychometricFnsAdjusted = psychometricFnsAdjusted;
%         
%         save("V1DamageData.mat", 'data');

    end

end

data.psychometricFns         = psychometricFns;
data.psychometricFnsDamaged  = psychometricFnsDamaged;
data.psychometricFnsAdjusted = psychometricFnsAdjusted;

save("V1DamageData2.mat", 'data');

%%

load("V1DamageData2.mat")
psychometricFns          = zeros(9, nItr, 21);
psychometricFnsDamaged   = zeros(9, nItr, 21);
psychometricFnsAdjusted  = zeros(9, nItr, 21);

psychometricFns(1:9, :, :)          = data.psychometricFns;
psychometricFnsDamaged(1:9, :, :)   = data.psychometricFnsDamaged;
psychometricFnsAdjusted(1:9, :, :)  = data.psychometricFnsAdjusted;

% load("V1DamageData2.mat")
% 
% psychometricFns(10:11, :, :)          = data.psychometricFns;
% psychometricFnsDamaged(10:11, :, :)   = data.psychometricFnsDamaged;
% psychometricFnsAdjusted(10:11, :, :)  = data.psychometricFnsAdjusted;

uniqStim            = unique(stimVector);
thresholds          = zeros(9, nItr);
thresholdsDamaged   = zeros(size(thresholds));
thresholdsAdjusted  = zeros(size(thresholds));

psychometricFnsFit         = zeros(9, nItr, 200);
psychometricFnsDamagedFit  = zeros(size(psychometricFnsFit));
psychometricFnsAdjustedFit = zeros(size(psychometricFnsFit));

% Calculate slopes
for i = 1:9
    for j = 1:nItr
        
        %%% Actual data
        x = rad2deg(uniqStim);
        y = squeeze( psychometricFns(i, j, :) )';
        
        % Psychometric function
        psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
        b0 = [90, std(x)]; % Initial guess
        
        lb = [-Inf, eps];
        ub = [ Inf, Inf];
        
        opts = optimoptions('lsqcurvefit','Display','off');
        bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);
        
        xFit = linspace(min(x), max(x), 200);
        yFit = psyFun(bHat, xFit);
        
        psychometricFnsFit(i, j, :) = yFit;
        thresholds(i, j) = bHat(2);

        %%% Damaged V1 data
        x = rad2deg(uniqStim);
        y = squeeze( psychometricFnsDamaged(i, j, :) )';
        
        % Psychometric function
        psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
        b0 = [90, std(x)]; % Initial guess
        
        lb = [-Inf, eps];
        ub = [ Inf, Inf];
        
        opts = optimoptions('lsqcurvefit','Display','off');
        bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);
        
        xFit = linspace(min(x), max(x), 200);
        yFit = psyFun(bHat, xFit);

        psychometricFnsDamagedFit(i, j, :) = yFit;
        thresholdsDamaged(i, j) = bHat(2);
        
        %%% Adjusted V1 data
        x = rad2deg(uniqStim);
        y = squeeze( psychometricFnsAdjusted(i, j, :) )';
        
        % Psychometric function
        psyFun = @(b,x) normcdf((x - b(1)) ./ b(2));
        b0 = [90, std(x)]; % Initial guess
        
        lb = [-Inf, eps];
        ub = [ Inf, Inf];
        
        opts = optimoptions('lsqcurvefit','Display','off');
        bHat = lsqcurvefit(psyFun, b0, x, y, lb, ub, opts);
        
        xFit = linspace(min(x), max(x), 200);
        yFit = psyFun(bHat, xFit);
        
        psychometricFnsAdjustedFit(i, j, :) = yFit;
        thresholdsAdjusted(i, j) = bHat(2);
    end
end


y1 = median(thresholds, 2);
y2 = median(thresholdsDamaged, 2);
y3 = median(thresholdsAdjusted, 2);

y1_std = mad(thresholds, 1, 2) ./ sqrt(size(thresholds,2)); % flag is 1 so it's wrt to median
y2_std = mad(thresholdsDamaged, 1, 2) ./ sqrt(size(thresholdsDamaged,2));
y3_std = mad(thresholdsAdjusted, 1, 2) ./ sqrt(size(thresholdsAdjusted,2));

% y1 = mean(thresholds, 2);
% y2 = mean(thresholdsDamaged, 2);
% y3 = mean(thresholdsAdjusted, 2);
% y1_std = std(thresholds, 0, 2) ./ sqrt(size(thresholds,2));
% y2_std = std(thresholdsDamaged, 0, 2) ./ sqrt(size(thresholdsDamaged,2));
% y3_std = std(thresholdsAdjusted, 0, 2) ./ sqrt(size(thresholdsAdjusted,2));


figure('Color','w'); hold on

lw = 1.8;
ms = 6;

errorbar(1:9, y1(1:9), y1_std(1:9), '-o', ...
    'LineWidth', lw, 'MarkerSize', ms, ...
    'CapSize', 8, 'DisplayName', 'Intact V1');

errorbar(1:9, y2(1:9), y2_std(1:9), '-s', ...
    'LineWidth', lw, 'MarkerSize', ms, ...
    'CapSize', 8, 'DisplayName', 'Damaged V1');

errorbar(1:9, y3(1:9), y3_std(1:9), '-^', ...
    'LineWidth', lw, 'MarkerSize', ms, ...
    'CapSize', 8, 'DisplayName', 'Adjusted V1');

hold off

ax = gca;
ax.FontSize = 12;
ax.LineWidth = 1.2;
ax.TickDir = 'out';
ax.Box = 'off';

xticks(1:9)
xticklabels({'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'})

xlabel('Proportion damaged','FontSize',13)
ylabel('Threshold (SD of psychometric fit)','FontSize',13)
set(gca,'YScale','log')

legend('Location','best','Box','off')

% exportgraphics(gcf,'Figure1.eps','ContentType','vector')

figure('Color','w','Position',[100 100 1400 700])

pDamagedList = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

lw_mean = 2;
lw_ind  = 0.5;
grayCol = [0.7 0.7 0.7];

for i = 1:2:9
    col = (i+1)/2;

    % ---------- Intact ----------
    ax1 = subplot(3,6,col); hold on
    for j = 1:nItr
        plot(xFit, squeeze(psychometricFnsFit(i,j,:)), ...
            'Color', grayCol, 'LineWidth', lw_ind)
    end
%     plot(xFit, mean(squeeze(psychometricFnsFit(i,:,:)),1), ...
%         'k','LineWidth',lw_mean)
    if col == 1
        ylabel('% CCW')
        title('Intact V1','FontWeight','bold')
    end
    hold off

    % ---------- Damaged ----------
    ax2 = subplot(3,6,6+col); hold on
    for j = 1:nItr
        plot(xFit, squeeze(psychometricFnsDamagedFit(i,j,:)), ...
            'Color', grayCol, 'LineWidth', lw_ind)
    end
%     plot(xFit, mean(squeeze(psychometricFnsDamagedFit(i,:,:)),1), ...
%         'k','LineWidth',lw_mean)
    if col == 1
        ylabel('% CCW')
        title(sprintf( 'Damaged V1 %.2f', pDamagedList(i)),'FontWeight','bold')
    else
        title(sprintf( '%.2f', pDamagedList(i)),'FontWeight','bold')
    end
    hold off

    % ---------- Adjusted ----------
    ax3 = subplot(3,6,12+col); hold on
    for j = 1:nItr
        plot(xFit, squeeze(psychometricFnsAdjustedFit(i,j,:)), ...
            'Color', grayCol, 'LineWidth', lw_ind)
    end
%     plot(xFit, mean(squeeze(psychometricFnsAdjustedFit(i,:,:)),1), ...
%         'k','LineWidth',lw_mean)
    if col == 1
        ylabel('% CCW')
        title('Adjusted V1','FontWeight','bold')
    end
    xlabel('Orientation')
    hold off

    % ---------- Axis standardization ----------
    set([ax1 ax2 ax3], ...
        'FontSize',10, ...
        'LineWidth',1, ...
        'Box','off', ...
        'TickDir','out', ...
        'YLim',[0 1])
end

% exportgraphics(gcf,'Figure2.eps','ContentType','vector')


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
tuningParams.beta(:) = lognrnd(2.5, 0.5, 1, nNeurons);   % Random values for beta parameter
tuningParams.eps1(:) = lognrnd(1, 0.8, 1, nNeurons);     % Random values for dynamic range control

end

function [tuningParamsV1Damaged] = getTuningParamsV1Damaged(tuningParams, damagedNrnIdxes)

% Neuron tuning parameters
tuningParamsV1Damaged.d = tuningParams.d;                      % Direction selectivity (fixed at 0)
tuningParamsV1Damaged.alpha = tuningParams.alpha;              % Aspect ratio (fixed at 2)
tuningParamsV1Damaged.b = tuningParams.b;                      % Controls sharpness of tuning curve (fixed at 2)
tuningParamsV1Damaged.q = tuningParams.q;                      % Controls amplitude of peak firing rate (variable)
tuningParamsV1Damaged.w = tuningParams.w;                      % Unused parameter (set to 1)
tuningParamsV1Damaged.UNTUNED_FILTER_AMPL = tuningParams.UNTUNED_FILTER_AMPL;  % Untuned filter amplitude (fixed at 0)
tuningParamsV1Damaged.eps1 = tuningParams.eps1;                % Controls dynamic range (variable)
tuningParamsV1Damaged.beta = tuningParams.beta;                % Controls dynamic range (variable)

% Set evoked activity zero for damaged neurons
tuningParamsV1Damaged.beta(damagedNrnIdxes) = 0;               % Evoked activity set to zero for damaged neuron

end

function [tuningParamsAdjusted] = getTuningParamsAdjusted(tuningParams, intactNrnIdxes)

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

function [spikes, decision] = decodeDecision(gainVector, trialIDx, timeBins, firingRates, ...
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

% Decision: CW (-1) or CCW (1) based on decoded orientation
decision = (thetaMLE > pi/2)*(-1) + (thetaMLE <= pi/2)*(1);

end
