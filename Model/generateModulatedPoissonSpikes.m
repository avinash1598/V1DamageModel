% ---------------------------------------------------------------------
% Function to generate spike trains using a modulated Poisson process.
% Given the time-dependent stimulus response of neurons, this function 
% simulates spike trains for each neuron over time based on a Poisson 
% spiking model. The firing rate is modulated by a gain factor for each 
% trial and neuron.
% 
% Inputs:
%    trlStimFRResponse : [nNeurons x timeBins] Matrix, the firing rate
%                        response of each neuron (for a single trial)
%                        over time in response to the stimulus.
%    trlGainVector     : [nNeurons x timeBins] Matrix, gain applied to
%                        the firing rates to simulate trial-by-trial
%                        variability in firing rates.
%    params            : Structure containing additional parameters:
%                        - timeStep: Time bin size in seconds (e.g., 0.001s)
%                        - timeBins: Vector of time bins (length = # time bins)
%                        - nNeurons: Number of neurons
%
% Outputs:
%    spikes            : [nNeurons x timeBins] Binary matrix of spike
%                        events. 1 indicates a spike at a given time bin
%                        for a given neuron, 0 indicates no spike.
%    modStimResponse   : [nNeurons x timeBins] The modulated stimulus
%                        response for each neuron over time. This is the
%                        product of the stimulus firing rate and the gain
%                        factor for each neuron.
%
% ---------------------------------------------------------------------
function [spikes, modStimResponse] = generateModulatedPoissonSpikes(trlStimFRResponse, trlGainVector, params)
    % Extract the parameters from the struct
    timeStep = params.timeStep;         % Size of each time bin (e.g., 0.001s)
    timeBins = params.timeBins;         % Vector of time bins
    nNeurons = params.nNeurons;         % Number of neurons

    % Compute the modulated stimulus response by multiplying the firing
    % rate with the trial-specific gain for each neuron. This simulates
    % variability in the neurons' firing rates across trials.
    modStimResponse = trlGainVector .* trlStimFRResponse; 
    
    % Calculate the probability of spiking in each time bin. The probability 
    % is derived by multiplying the modulated firing rate by the time step.
    % This ensures that the firing probability over the time window remains 
    % consistent with the desired firing rate (rate * dt gives the probability 
    % for a Poisson process). Note: This way of calculating probability
    % makes an assumption that within a given interval chances of occuring
    % two spikes are really really low.
    probSpk = modStimResponse .* timeStep;
    
    % Generate random numbers for each neuron and time bin. If the random number
    % is less than the calculated spiking probability, a spike (1) is generated.
    % This is based on the assumption that in very small time bins, the probability
    % of more than one spike in a bin is negligible.
    rndNumbers = rand(nNeurons, length(timeBins));  % Random numbers in the range [0, 1]
    
    % Generate spike matrix by comparing random numbers with the spiking probability.
    % If the random number is less than the spiking probability, the neuron fires (1).
    spikes = rndNumbers < probSpk;
end


% % Single trial spike generator for modulated posisson process
% function [spikes, modStimResponse] = generateModulatedPoissonSpikes(trlStimFRResponse, trlGainVector, params)
%     timeStep = params.timeStep;
%     timeBins = params.timeBins;
%     nNeurons = params.nNeurons;
% 
%     modStimResponse = trlGainVector.*trlStimFRResponse; 
%     % Note: this way of calculating probability of spiking in each time bin
%     % relies on the assumption that there exist a very small time bin dt within
%     % which no two spikes can be fired. Hence, firing rate of the neurons
%     % should not be very great compared to the size of the time bin.
%     probSpk = modStimResponse.*timeStep;
%     rndNumbers = rand(nNeurons, length(timeBins)); % Generate for each trial and each time bin
%     spikes = rndNumbers < probSpk;
% end