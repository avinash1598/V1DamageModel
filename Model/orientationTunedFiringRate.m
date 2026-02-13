function [FR] = orientationTunedFiringRate(orientations, neuronsPrefOrientations, params)
    % ORIENTATIONTUNEDFIRINGRATE Computes the firing rates of neurons based on their 
    % preferred orientation using a tuning function model.
    %
    % INPUTS:
    % orientations - (nTrials x 1 vector) Array of stimulus orientations (in radians).
    %                These are the orientations that the subject is exposed to.
    %
    % neuronsPrefOrientations - (1 x nNeurons vector) Array of preferred orientations 
    %                           for each neuron (in radians). Each neuron is tuned to 
    %                           a specific orientation.
    %
    % params - (struct) Structure containing the tuning function parameters:
    %   q    : Non-linearity exponent applied to the response of the neuron.
    %   eps1 : Baseline (offset) firing rate for each neuron.
    %   beta : Controls the range and scaling of the firing rates.
    %
    % OUTPUT:
    % FR - (nTrials x nNeurons matrix) Firing rates of neurons in response to the given 
    %      stimulus orientations. Each row represents a trial (stimulus orientation), 
    %      and each column represents a neuron.
    %
    % DESCRIPTION:
    % The function computes the orientation-tuned firing rate of each neuron. This is
    % done by first calculating the unnormalized tuned response of each neuron to the 
    % given orientations, followed by normalization, and applying a power law 
    % non-linearity to adjust for firing rates.

    % Step 1: Get unnormalized tuning response of neurons based on their preferred orientation.
    unnormalizedResp = getUnormalizedTunedResponse(orientations, neuronsPrefOrientations, params);
    
    % Step 2: Normalize the combined channel output (tuning curve of each neuron)
    %         between 0 and 1 based on its response to a set of 180 different 
    %         orientations (from 0 to pi).
    tuningFnOrientations = linspace(0, pi, 180);  % Set of orientations (0 to pi).
    unormalizedTuningFn = getUnormalizedTunedResponse(tuningFnOrientations, neuronsPrefOrientations, params);
    
    % Normalize the unnormalized tuning function between 0 and 1.
    minVals = min(unormalizedTuningFn);  % Minimum firing rate across all orientations (for each neuron).
    maxVals = max(unormalizedTuningFn);  % Maximum firing rate across all orientations (for each neuron).
    combinedChannelOp = (unnormalizedResp - minVals) ./ (maxVals - minVals);  % Normalize for each neuron.
    
    % Step 3: Compute the final firing rate using power law non-linearity.
    % This models how the response of the neuron changes based on non-linear scaling.
    q = params.q;        % Non-linearity exponent
    eps1 = params.eps1;  % Baseline firing rate (offset)
    beta = params.beta;  % Scaling factor
    
    % Apply non-linearity to normalized responses
    FR = eps1 + beta.*(combinedChannelOp.^q);
end

function [FR] = getUnormalizedTunedResponse(orientations, neuronsPrefOrientations, params)
    % GETUNORMALIZEDTUNEDRESPONSE Computes the raw (unnormalized) tuning function response
    % of neurons based on the stimulus orientations and their preferred orientations.
    %
    % INPUTS:
    % orientations - (nTrials x 1 vector) Array of stimulus orientations (in radians).
    %
    % neuronsPrefOrientations - (1 x nNeurons vector) Preferred orientations for each neuron (in radians).
    %
    % params - (struct) Structure containing the tuning function parameters:
    %   d    : Direction selectivity of neurons. (Set to zero in this case, no direction selectivity.)
    %   alpha: Aspect ratio. Controls the sharpness of the tuning curve.
    %   b    : Sharpness of the tuning curve (power of exponent).
    %   q    : Non-linearity exponent applied after tuning curve.
    %   w    : Weight of untuned filter amplitude.
    %   UNTUNED_FILTER_AMPL : Amplitude of untuned filter response (set to zero in this case).
    %
    % OUTPUT:
    % FR - (nTrials x nNeurons matrix) Unnormalized firing rate responses for each trial (orientation)
    %      and each neuron (based on its preferred orientation).
    %
    % DESCRIPTION:
    % This function calculates the raw, unnormalized firing rate of neurons based on their 
    % orientation tuning curves. It incorporates factors like direction selectivity, sharpness 
    % (alpha and b), and untuned responses (if applicable). The result is an unrectified response,
    % which will later be normalized and modified by non-linearity.
    
    % Extract parameters from the input struct
    d = params.d;               % Direction selectivity
    alpha = params.alpha;       % Aspect ratio, controls the sharpness of the curve.
    b = params.b;               % Controls the sharpness of the tuning curve.
    w = params.w;               % Weight of untuned filter
    untunedFltAmp = params.UNTUNED_FILTER_AMPL;  % Untuned filter amplitude.
    
    % Step 1: Compute the cosine of the angular difference between stimulus orientation and 
    % the neuron's preferred orientation.
    t1 = orientations' - neuronsPrefOrientations;  % Angular difference (nTrials x nNeurons)
    
    % Step 2: Apply direction selectivity (if non-zero, modifies the response).
    t2 = 1 + 0.5 * (sign(cos(t1)) - 1) .* d;  % Direction-selective factor
    
    % Step 3: Compute the tuning curve response.
    % This uses a combination of cosine tuning and exponential factors to simulate neuron selectivity.
    t3 = cos(t1) .* exp(-0.5 * (cos(t1).^2) .* (1 - alpha.^2) ).^b;  % Tuning response
    
    % Step 4: Compute the tuned and untuned filter responses.
    rTuned = t2 .* t3;                % Tuned linear filter response
    rUntuned = w * untunedFltAmp;     % Untuned filter response (zero in this case)
    
    % Step 5: Combine the tuned and untuned responses.
    combinedChOp = rTuned - rUntuned;  % Overall response
    
    % Step 6: Apply rectifier (set negative values to zero).
    combinedChOp(combinedChOp < 0) = 0;
    
    % Step 7: Return the final unnormalized firing rate response.
    FR = combinedChOp;
end



% function [FR] = orientationTunedFiringRate(orientations, neuronsPrefOrientations, params)
%     % TUNING_FN_FR: Tuning function of all the neurons used for
%     % normalization
%     % Extract parameters from the structure
%     q = params.q;
%     eps1 = params.eps1;
%     beta = params.beta;  
% 
%     unnormalizedResp = getUnormalizedTunedResponse(orientations, neuronsPrefOrientations, params);
% 
%     % Normalize combined chanel output between zero and one for each neuron
%     % Once the tuning curves are normalized, adjust the amplitudes to set
%     % baseline and peak firing rates of individual neurons.
%     % COMBINED_CHANNEL_OUTPUT : ntrials * nNeurons
%     % Note: For normalization, unnormalized tuning function is required
%     % i.e. firing rates of all the neurons at all possible orientations.
%     tuningFnOrientations = linspace(0, pi, 180);
%     unormalizedTuningFn = getUnormalizedTunedResponse(tuningFnOrientations, neuronsPrefOrientations, params);
%     minVals = min(unormalizedTuningFn);  % Minimum value of each column
%     maxVals = max(unormalizedTuningFn);  % Maximum value of each column
%     combinedChannelOp = (unnormalizedResp - minVals) ./ (maxVals - minVals);
% 
%     % Compute firign rate using power law non-linearity
%     FR = eps1 + beta.*(combinedChannelOp.^q);
% end
% 
% function [FR] = getUnormalizedTunedResponse(orientations, neuronsPrefOrientations, params)
%     % Extract parameters from the structure
%     d = params.d;
%     alpha = params.alpha;
%     b = params.b;
%     q = params.q;
%     w = params.w;
%     untunedFltAmp = params.UNTUNED_FILTER_AMPL;
% 
%     % TODO: Add a check to make sure all these parameters are provided
% 
%     % Mean firing rates profiles of all the neurons from 
%     % the given input stimuli.
%     t1 = orientations' - neuronsPrefOrientations;
%     t2 = 1 + 0.5*( sign(cos(t1)) - 1).*d;
%     t3 = cos(t1).*exp(-0.5*(cos(t1).^2).*(1 - alpha.^2) ).^b;
% 
%     rTuned = t2.*t3; % Tuned linear filter response
%     rUntuned = w*untunedFltAmp; % Untuned filter response
%     combinedChOp = rTuned - rUntuned;
%     % TODO: Normalization leave it for later: Ask Robbe
%     % Apply rectifier to the linear filter output
%     combinedChOp(combinedChOp < 0) = 0;
% 
%     FR = combinedChOp;
% end