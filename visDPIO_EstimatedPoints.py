import os
from scipy.io import loadmat

def load_all_mat_files(directory="."):
    """
    Load all .mat files from the specified directory and store the data in a dictionary.
    
    Returns:
        all_data: A dictionary containing the data from all .mat files.
    """
    data_files = [f for f in os.listdir(directory) if f.endswith('.mat')]
    
    all_data = {}

    for file in data_files:
        # Load the .mat file
        file_path = os.path.join(directory, file)
        data = loadmat(file_path)
        
        # Extract the subject name from the filename
        subjN = file.replace('estData', '').replace('.mat', '')
        
        # Prepare the data from the .mat file
        all_data[subjN] = {}
        
        # Load the fit results from the .mat file
        for i in range(4):  # Assuming four datasets were stored in the .mat files
            key = f'fit_results_{i}'
            if key in data:
                fit_data = data[key]
                
                # Store all the relevant details in the dictionary
                all_data[subjN][key] = {
                    'fitted_polynomial': fit_data['fitted_polynomial'][0],
                    'max_slope': fit_data['max_slope'][0][0],
                    'L2_at_max_slope': fit_data['L2_at_max_slope'][0][0],
                    'OAE_level_at_max_slope': fit_data['OAE_level_at_max_slope'][0][0],
                    'L2_half_max_slope': fit_data['L2_half_max_slope'][0][0],
                    'OAE_level_half_max_slope': fit_data['OAE_level_half_max_slope'][0][0],
                }

    return all_data

# Example usage
all_data = load_all_mat_files(directory="Estimace/")  # Provide the path where .mat files are located


