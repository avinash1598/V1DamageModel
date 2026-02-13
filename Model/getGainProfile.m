function gainProfiles = getGainProfile(nNeurons, timebins, stimDur)
    % This function generates gain profiles for a given number of neurons
    % over specified time bins, based on cumulative Gaussian distributions.
    % These gain profiles are used to modulate top-down gain (bias due to 
    % decision in previous trials) of neurons.
    %
    % Inputs:
    %   nNeurons: Number of neurons to generate gain profiles for
    %   timebins: Time bins for the stimulus duration
    %   stimDur: Total stimulus duration
    %
    % Output:
    %   gainProfiles: A matrix (nNeurons x length(timebins)) containing the
    %   cumulative Gaussian distribution values for each neuron over time.
    
    x = linspace(0, stimDur, length(timebins));
    cumulativeGaussians = zeros(nNeurons, length(x));
    means = 0.4 + 0.1*randn(1, nNeurons);
    stdDevs = 0.08*rand(1, nNeurons);

    for i = 1:nNeurons
        % Calculate the cumulative Gaussian distribution for neuron i
        % using its unique mean and standard deviation
        cumulativeGaussians(i, :) = normcdf(x, means(i), stdDevs(i));
    end

    gainProfiles = cumulativeGaussians;
end


% function gainProfiles = getGainProfile(nNeurons, timebins, stimDur)
%     x = linspace(0, stimDur, length(timebins));
%     cumulativeGaussians = zeros(nNeurons, length(x));
% 
%     % Parameters adjusted so that cum distribution is between 0 and 1
%     means = 0.4 + 0.1*randn(1, nNeurons);   % Random means from normal distribution
%     stdDevs = 0.08*rand(1, nNeurons); % Random std devs (from uniform, shifted to avoid near-zero values)
% 
%     for i = 1:nNeurons
%         cumulativeGaussians(i, :) = normcdf(x, means(i), stdDevs(i));
%     end
% 
%     gainProfiles = cumulativeGaussians;
% end