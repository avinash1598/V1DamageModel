function [retData] = getLLhChoice_CASANDRE(stimVals, modelParams)
    
    sigma_d     =   modelParams.sigma_d;
    Cd          =   modelParams.Cd;
    Cc          =   modelParams.Cc;
    sigma_m     =   modelParams.sigma_m;
    sampleCnt   =   1000;
    % sampleCnt   =   modelParams.sampleCnt;
    
    choicePDFs = zeros(4, numel(stimVals));
    
    for sidx = 1:numel(stimVals)
        stimVal = stimVals(sidx);
    
        % Uniformly sample lognormal cdf to obtain estimated of sigma_d evenly
        % tiling the distribution. TODO: try weighted averaging but probabilities
        % around each sample is same, so this step can just be omitted
        muLogN    = log((sigma_d.^2)./sqrt(sigma_m.^2 + sigma_d.^2));
        sigmaLogN = sqrt(log((sigma_m.^2)./(sigma_d.^2) + 1));
        sigma_d_hat_samples  = logninv(linspace(1/sampleCnt, 1 - 1/sampleCnt, sampleCnt), muLogN, sigmaLogN);
        
        % Mean and var of confidence variable
        mu_c = (stimVal - Cd) ./ sigma_d_hat_samples;
        sigma_c = sigma_d ./ sigma_d_hat_samples;
        
        % These three CDF values will be used to compute the probabilitites
        x = [-Cc, 0, Cc];
        x1 = repmat(x, [sampleCnt 1]);
        x2 = repmat(mu_c', [1 numel(x)]);
        x3 = repmat(sigma_c', [1 numel(x)]);
        cdf_vals = normcdf(x1, x2, x3);
        
        % TODO: weight cdf_vals by corresponding probabilitites. This probably
        % can be omitted since probabilitites around each sample is the same.
        mean_cdf_vals = mean(cdf_vals, 1);
        
        p_c1_d0 = mean_cdf_vals(1); % CW, HC
        p_c0_d0 = mean_cdf_vals(2) - mean_cdf_vals(1); % CW, LC
        p_c1_d1 = 1 - mean_cdf_vals(3); % CCW, HC
        p_c0_d1 = mean_cdf_vals(3) - mean_cdf_vals(2); % CCW, LC
        
        choicePDFs(1, sidx) = p_c1_d0; % CW, HC
        choicePDFs(2, sidx) = p_c0_d0; % CW, LC
        choicePDFs(3, sidx) = p_c1_d1; % CCW, HC
        choicePDFs(4, sidx) = p_c0_d1; % CCW, LC
    end
    
    retData = choicePDFs;
end
