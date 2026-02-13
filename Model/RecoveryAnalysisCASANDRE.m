clear all
close all

% Parameters
mu_d = linspace(70, 110, 101);   % Make it an array - Mean of decision variable Vd (e.g., stimulus value)
sigma_d = 6;                     % True stimulus uncertainty
sigma_m = 2;                     % Meta-uncertainty (std of log-normal noise)
Cd = 90;                         % Decision criterion
Cc = 1.0;                        % Confidence criterion
n_trials = 200;                  % Number of simulated trials

% Choice counts: Order - (CW, HC), (CW, LC), (CCW, HC), (CCW, LC)
nChoice = zeros(4, numel(mu_d));

% Simulate trials

for sidx=1:length(mu_d)

    stim = mu_d(sidx);

    for t = 1:n_trials
        % First stage: simulate decision variable (normal distribution)
        Vd = normrnd(stim, sigma_d);
        
        % Second stage: simulate noisy estimate of sigma_d (log-normal)
        mu_log = log(sigma_d^2 / sqrt(sigma_m^2 + sigma_d^2));  % log-mean
        sigma_log = sqrt(log(1 + (sigma_m^2 / sigma_d^2)));     % log-std
        sigma_d_hat = lognrnd(mu_log, sigma_log);
        
        % Make choice
        decision = Vd > Cd;
        
        % Compute confidence variable (Vc)
        Vc = abs(Vd - Cd) / sigma_d_hat;
    
        % Generate binary confidence report
        conf = Vc > Cc;

        % (CW, HC), (CW, LC), (CCW, HC), (CCW, LC)
        if decision == 0 && conf == 1
            nChoice(1, sidx) = nChoice(1, sidx) + 1;
        elseif decision == 0 && conf == 0
            nChoice(2, sidx) = nChoice(2, sidx) + 1;
        elseif decision == 1 && conf == 1
            nChoice(3, sidx) = nChoice(3, sidx) + 1;
        elseif decision == 1 && conf == 0
            nChoice(4, sidx) = nChoice(4, sidx) + 1;
        end
    end
end

stimVals    = mu_d;
optParams   = optimizeCASANDRE(stimVals, nChoice);

%% Analytical solution
modelParams.sigma_d = optParams(1);
modelParams.Cd      = 90;
modelParams.Cc      = optParams(2);
modelParams.sigma_m = optParams(3);

choicePDFs  = getLLhChoice_CASANDRE(stimVals, modelParams);


figure

subplot(2, 3, 1)
hold on
scatter(mu_d, nChoice(1, :), DisplayName='CW, HC', HandleVisibility='off')
scatter(mu_d, nChoice(2, :), DisplayName='CW, LC', HandleVisibility='off')
scatter(mu_d, nChoice(3, :), DisplayName='CCW, HC', HandleVisibility='off')
scatter(mu_d, nChoice(4, :), DisplayName='CCW, LC', HandleVisibility='off')

plot(stimVals, choicePDFs(1, :)*n_trials, DisplayName='CW, HC', LineWidth=1.5)
plot(stimVals, choicePDFs(2, :)*n_trials, DisplayName='CW, LC', LineWidth=1.5)
plot(stimVals, choicePDFs(3, :)*n_trials, DisplayName='CCW, HC', LineWidth=1.5)
plot(stimVals, choicePDFs(4, :)*n_trials, DisplayName='CCW, LC', LineWidth=1.5)

hold off
xlabel("orientation")
ylabel("choice count")
legend

