import os
import re

def create_dicts(folder_path):
    # Define the pattern to match the date and time in the filenames
    date_time_pattern = re.compile(r'(\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})')

    # Create dictionaries to store unique date and time strings and their corresponding filenames for '_L_' and '_R_'
    date_time_filenames_dict_l = {}
    date_time_filenames_dict_r = {}

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if 'CMclickOAE' in filename:
            # Strip anything after '_L_' or '_R_'
            filename = re.sub(r'(_[LR]_).*', r'\1', filename)
            match = date_time_pattern.search(filename)
            if match:
                date_time = match.group(1)
                lc_value = re.search(r'_Lc_(\d+)dB', filename)
                if lc_value:
                    lc_value = int(lc_value.group(1))
                else:
                    lc_value = 0  # Set default value if 'Lc_' is not found
                if "_L_" in filename:
                    if date_time not in date_time_filenames_dict_l:
                        date_time_filenames_dict_l[date_time] = set()
                    date_time_filenames_dict_l[date_time].add((lc_value, filename))
                elif "_R_" in filename:
                    if date_time not in date_time_filenames_dict_r:
                        date_time_filenames_dict_r[date_time] = set()
                    date_time_filenames_dict_r[date_time].add((lc_value, filename))

    # Sort values by the lc_value
    for date_time, filenames in date_time_filenames_dict_l.items():
        date_time_filenames_dict_l[date_time] = [filename for _, filename in sorted(filenames)]
    for date_time, filenames in date_time_filenames_dict_r.items():
        date_time_filenames_dict_r[date_time] = [filename for _, filename in sorted(filenames)]

    return date_time_filenames_dict_l, date_time_filenames_dict_r

def print_dicts_as_command_line(folder_path):
    data_dict_l, data_dict_r = create_dicts(folder_path)
    dLsorted = sorted(data_dict_l.values(), key=lambda x: int(re.search(r'_Lc_(\d+)dB', x[0]).group(1)))
    dRsorted = sorted(data_dict_r.values(), key=lambda x: int(re.search(r'_Lc_(\d+)dB', x[0]).group(1)))
    #Lunlist = [for prvek in  if isinstance(prvek,'list')
    
    dLsorted.insert(0,folder_path)
    dRsorted.insert(0,folder_path)
    
    dL = [prvek[0] if isinstance(prvek,list) else prvek for prvek in dLsorted]
    dR = [prvek[0] if isinstance(prvek,list) else prvek for prvek in dRsorted]
    
    print(f"{dL}")
    print(f"{dR}")
    
# Specify the folder containing the files
folder_path = "Results/s089/"

# Print the dictionaries as command line output
print_dicts_as_command_line(folder_path)



# Extract values and sort them based on the integer value after '_Lc_'
#s002R_values_sorted = sorted(dL.values(), key=lambda x: int(re.search(r'_Lc_(\d+)dB', x[0]).group(1)))

# Convert into a new dictionary with a single key 's002R'
#s002R_ordered = {'s002R': s002R_values_sorted}

