function expData = loadExpData(filename)
    % Function to load expData and apply custom reformatting
    % Inputs:
    %   - filename: The name of the .mat file to load (e.g., 'expData.mat')
    %   - reformatOptions: A struct containing options to reformat the data
    % Output:
    %   - expData: The loaded and optionally reformatted data
    
    % Load the data
    loadedData = load(filename);
    
    % Check if expData exists in the loaded file
    if ~isfield(loadedData, 'expData')
        error('expData structure not found in the file.');
    end
    
    expData = loadedData.expData;
    
    % Convert logical trialResponses matrix to double 
    % expData.trialResponses = double(expData.trialResponses);  % Convert logical matrix to double
    
    % Don't convert logical trialResponses to double since it blows up the
    % memory used.
    expData.trialResponses = expData.trialResponses;
end
