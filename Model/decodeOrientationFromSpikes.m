% ---------------------------------------------------------------------
% Function to decode stimulus orientation from spike data using a Bayesian approach.
% This function performs decoding based on the neuron's tuning curves and 
% the observed spike counts, estimating the most likely stimulus orientation.

% INPUTS:
% - spikes:            [nNeurons x nTimeBins] Binary matrix containing the spike responses for all neurons over time.
% - neuronsPrefOrientation: [1 x nNeurons] Preferred orientations of the neurons (in radians), determining their tuning.
% - params:            Structure containing several parameters related to the time bins, number of neurons, and stimulus duration.
%   - params.timeBins: [1 x nTimeBins] Time bin vector representing the time points for spike recording.
%   - params.nNeurons: Scalar representing the total number of neurons.
%   - params.stimDuration: Scalar representing the total duration of stimulus presentation (in seconds).
% - tuningParams:      Structure containing tuning curve parameters for the population of neurons (used for computing firing rates based on orientation).
%   - tuningParams: Includes various fields such as alpha, q, beta, etc., which control the neuron tuning properties.

% OUTPUTS:
% - thetaMLE: [1 x numIntervals] The maximum likelihood estimate (MLE) of the stimulus orientation for each time window.
% ---------------------------------------------------------------------
function [thetaMLE, pdfData] = decodeOrientationFromSpikes(spikes, neuronsPrefOrientation, params, tuningParams)

    % Extract relevant parameters from the input 'params' structure
    % timeBins = params.timeBins;             % Time bins for the spike train
    % nNeurons = params.nNeurons;             % Number of neurons in the population    
    stimDuration = params.stimDuration;     % Stimulus duration (total time)
    
    % Create a vector of possible orientations (0 to pi) for decoding
    decThetas = linspace(0, pi, 360);                     % Candidate stimulus orientations (360 values)
    
    % Compute the tuning function firing rates for each candidate orientation
    % Tuning function computes firing rates as a function of orientation and neuron preferences
    tuningFnFR = orientationTunedFiringRate(decThetas, neuronsPrefOrientation, tuningParams);
    
    % Decode orientation from aggrgate spike count (no need to consider small individual time windows)
    aggregateSpikeCounts = sum(spikes, 2);
    
    n = aggregateSpikeCounts;     % Spike count vector for all neurons in the current window
    f_theta = tuningFnFR';        % Firing rates for all neurons for each candidate orientation

    % Add a small epsilon to the firing rates to avoid numerical issues (log of 0)
    temp = stimDuration * f_theta + eps; 
    log_pdf = sum(n .* log(temp), 1) - sum(temp, 1) - sum(log(factorial(n)), 1); 

    [max_val, idx_max] = max(log_pdf);  % Index of the orientation that maximizes log_pdf
    thetaMLE =  decThetas(idx_max);    % Store the corresponding orientation (MLE)
    
    % convert log PDF to normalized PDF
    log_pdf = log_pdf - max(log_pdf);
    pdf = exp(log_pdf);
    x_deg = rad2deg(decThetas);
    dx = x_deg(2) - x_deg(1);
    pdf = pdf./sum(pdf*dx, 'all');

    % Calculate mean and variance
    mu    = sum( x_deg.*pdf*dx );
    sigma = sqrt( sum( (x_deg - mu).^2.*pdf*dx ) );

    pdfData.x_deg   = x_deg;
    pdfData.log_pdf = log_pdf;
    pdfData.pdf     = pdf;
    pdfData.mu      = mu;
    pdfData.sigma   = sigma;
    
    if sigma == 0
        keyboard
    end
    % Return both MLE estimate and PDF
    % Does log pdf needs to be converted to normal pdf and normalized
end



% function [thetaMLE] = decodeOrientationFromSpikes(spikes, neuronsPrefOrientation, params, tuningParams)
% 
%     % Extract relevant parameters from the input 'params' structure
%     timeBins = params.timeBins;             % Time bins for the spike train
%     nNeurons = params.nNeurons;             % Number of neurons in the population    
%     stimDuration = params.stimDuration;     % Stimulus duration (total time)
% 
%     % Decoding parameters:
%     tauStep = 0.02;                         % Step size of the decoding window (20ms)
%     numIntervals = floor(stimDuration / tauStep); % Total number of decoding windows (20ms steps)
%     intervalSize = floor(length(timeBins) / numIntervals); % Number of time bins per decoding window
%     
%     % Initialize arrays for storing aggregate spike counts, tau values, and MLE estimates
%     aggregateSpikeCounts = zeros(nNeurons, numIntervals); % Spike counts per neuron for each decoding window
%     taus = zeros(1, numIntervals);                        % Stores time points for each decoding window
%     thetaMLE = zeros(1, numIntervals);                    % Stores the decoded orientation (MLE) for each window
%     
%     % Create a vector of possible orientations (0 to pi) for decoding
%     decThetas = linspace(0, pi, 360);                     % Candidate stimulus orientations (360 values)
%     
%     % Compute the tuning function firing rates for each candidate orientation
%     % Tuning function computes firing rates as a function of orientation and neuron preferences
%     tuningFnFR = orientationTunedFiringRate(decThetas, neuronsPrefOrientation, tuningParams);
%     
%     % STEP 1: Bin spike counts into coarser intervals for decoding windows
%     % This step helps reduce the dimensionality by summing spike counts over larger time windows.
%     for i = 1:numIntervals
%         % Define the time window (starting from 1 to end to sum spike counts over time)
%         startCol = 1; % Start from the first time bin - start is always 1 - window size increases with time
%         endCol = min(i * intervalSize, length(timeBins)); % End at the current decoding window
%         
%         % Assign the current time window to 'taus' (used for tracking time steps)
%         taus(i) = i*tauStep; % Tau value for the current decoding window
%     
%         % Sum spike counts over the current time window (for all neurons)
%         intervalData = sum(spikes(:, startCol:endCol), 2);
%         aggregateSpikeCounts(:, i) = intervalData; % Store the result in aggregateSpikeCounts
%     end
%     
%     % STEP 2: Perform Bayesian decoding for each time window
%     % For each time window, estimate the stimulus orientation (theta) based on the
%     % observed spike counts and the tuning function firing rates.
%     
%     for t = 1:size(aggregateSpikeCounts, 2) % Iterate over each time window
%         n = aggregateSpikeCounts(:, t);     % Spike count vector for all neurons in the current window
%         f_theta = tuningFnFR';              % Firing rates for all neurons for each candidate orientation
%         
%         % STEP 3: Compute the log-likelihood (log PDF) for each orientation
%         % The goal is to find the orientation that maximizes the likelihood given the spike counts.
%         
%         % Add a small epsilon to the firing rates to avoid numerical issues (log of 0)
%         temp = taus(t) * f_theta + eps; 
%         
%         % Calculate the log-likelihood for each orientation (log-PDF)
%         % log_pdf computes the log of the probability of observing 'n' spikes given each candidate orientation.
%         log_pdf = sum(n .* log(temp), 1) - sum(temp, 1) - sum(log(factorial(n)), 1); 
%         % pdf = exp(log_pdf); % Just don't convert to exp to avoid values from blowing up! But averaging pdf won't work with log pdfs
%         % normalized_pdf = pdf / sum(pdf);
% 
%         % Find the orientation that maximizes the log-likelihood (MLE)
%         [max_val, idx_max] = max(log_pdf);  % Index of the orientation that maximizes log_pdf
%         thetaMLE(t) =  decThetas(idx_max);  % Store the corresponding orientation (MLE)
%     end
% end
% 


% function [thetaMLE] = decodeOrientationFromSpikes(spikes, neuronsPrefOrientation, params, tuningParams)
%     timeBins = params.timeBins;
%     nNeurons = params.nNeurons;    
%     stimDuration = params.stimDuration;
% 
%     % Decoding params
%     tauStep = 0.02; % Increment decoding window by 20 ms
%     numIntervals = floor(stimDuration / tauStep);
%     intervalSize = floor(length(timeBins) / numIntervals); 
% 
%     aggregateSpikeCounts = zeros(nNeurons, numIntervals);
%     taus = zeros(1, numIntervals);
%     thetaMLE = zeros(1, numIntervals);
%     % avg_pdf = zeros(1, size(decThetas, 2));
% 
%     % Bayesian decoding on the firing maps to decode stimulus direction
%     % Decode for each trial separately first
%     decThetas = linspace(0, pi, 360);
%     tuningFnFR = orientationTunedFiringRate(decThetas, neuronsPrefOrientation, tuningParams);
% 
%     % Do I need to consider the time dependent response as well! Let's see if
%     % decoding works just with tuning function without any knowledge of time
%     % dependent response (response profile is same for all so should not make much difference)
%     % Bin spike counts into coarser intervals
% 
%     % Get aggregate spike count of each neuron in the given decoding window
%     for i = 1:numIntervals
%         % Increasing temporal windows
%         startCol = 1; %(i-1) * intervalSize + 1; %1
%         endCol = min(i * intervalSize, length(timeBins));  % Handle the last interval
%         taus(i) = i*tauStep; % TODO: set tau as well
% 
%         intervalData = sum(spikes(:, startCol:endCol), 2);
%         aggregateSpikeCounts(:, i) = intervalData;
%     end
% 
%     % iterate for spike count in each time bin
%     for t=1:size(aggregateSpikeCounts, 2)
%         n = aggregateSpikeCounts(:, t);
%         f_theta = tuningFnFR';
% 
%         % Probability desnity function estimation for theta distribution
%         temp = taus(t)*f_theta + eps; % Adding a very small noise to avoid log going to NaN
%         log_pdf = sum(n.*log(temp), 1) - sum(temp, 1) - sum(log(factorial(n)), 1); % Fixed error. Bingo! now decoding works! Factorial is pushing the value really high
%         % pdf = exp(log_pdf); % Just don't convert to exp to avoid values from blowing up! But averaging pdf won't work with log pdfs
%         % normalized_pdf = pdf / sum(pdf);
% 
%         [max_val, idx_max] = max(log_pdf); %max(normalized_pdf);
%         thetaMLE(t) =  decThetas(idx_max);
%     end
% end

