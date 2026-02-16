function optParams = optimizeCASANDRE(stimVals, nChoice)

% stimVals in degree

addpath('C:\Users\avinash1598\Desktop\V1DamageModel\V1DamageModel\Model\Scripts\')

metaData.stimVals   = stimVals;
metaData.nChoice    = nChoice;
nStarts = 30;
nParams = 3;

x_all = zeros(nStarts, nParams);
f_all = zeros(nStarts, 1);

for itr = 1:nStarts
    
    success = false;
    
    while ~success
        try
            sigma_d     =   rand;
            Cc          =   rand;
            sigma_m     =   rand;
            
            params = [sigma_d Cc sigma_m];
            objFun = @(x) computeNLL(x, metaData);

            lb = zeros(size(params));     % same as before
            ub = [];                      % example finite upper bounds
            
            options = optimoptions('fmincon', ...
                'Display', 'iter', ...
                'Algorithm', 'sqp', ...          
                'MaxIterations', 1000, ...
                'MaxFunctionEvaluations', 20000);
            
            x0 = params;   % Initial guess (required for fmincon)
            
            [optimalValues, fval, exitflag, output] = fmincon(objFun, x0, ...
                [], [], [], [], lb, ub, [], options);
            
            disp(exitflag)
            disp(output.firstorderopt)
            
            if exitflag <= 0
                error('fminconn failed: %s', output.message)
            end
            
            x_all(itr, :) = optimalValues;
            f_all(itr)    = fval;
            
            success = true;

        catch ME
            disp(ME)
        end
    end
end

[~, idx]  = min(f_all);
optParams = x_all(idx, :);

end


% Loss function for optimization
function nll = computeNLL(params, metaData)

stimVals            = metaData.stimVals;
nChoice             = metaData.nChoice;

modelParams.sigma_d = params(1);
modelParams.Cd      = 90;
modelParams.Cc      = params(2);
modelParams.sigma_m = params(3);

choicePDFs  = getLLhChoice_CASANDRE(stimVals, modelParams);
choicePDFs  = choicePDFs + eps;

% NLL
nll = - sum( sum( nChoice.* log( choicePDFs ) ) );

end


