import os
import re

def create_dicts(folder_path):
    # Define the pattern to match the date, time, and F2b in the filenames
    date_time_f2b_pattern = re.compile(r'(\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_F2b_\d+Hz)')

    # Create dictionaries to store unique date_time_f2b strings and their corresponding filenames for '_L_' and '_R_' based on L2 value
    date_time_filenames_dict_l = {}
    date_time_filenames_dict_r = {}

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if 'p4swDPOAE' in filename:
            match = date_time_f2b_pattern.search(filename)
            if match:
                date_time_f2b = match.group(1)
                l2_value_match = re.search(r'L2_(\d+)dB', filename)
                if l2_value_match:
                    l2_value = int(l2_value_match.group(1))
                else:
                    l2_value = 0  # Set default value if 'L2_' is not found

                if "_L_" in filename:
                    if l2_value not in date_time_filenames_dict_l:
                        date_time_filenames_dict_l[l2_value] = set()
                    date_time_filenames_dict_l[l2_value].add(date_time_f2b)
                elif "_R_" in filename:
                    if l2_value not in date_time_filenames_dict_r:
                        date_time_filenames_dict_r[l2_value] = set()
                    date_time_filenames_dict_r[l2_value].add(date_time_f2b)

    # Convert sets to sorted lists and add folder path as the first element
    date_time_filenames_dict_l = {k: [folder_path] + sorted(list(v)) for k, v in date_time_filenames_dict_l.items()}
    date_time_filenames_dict_r = {k: [folder_path] + sorted(list(v)) for k, v in date_time_filenames_dict_r.items()}

    return date_time_filenames_dict_l, date_time_filenames_dict_r

def print_dicts_as_command_line(folder_path):
    data_dict_l, data_dict_r = create_dicts(folder_path)
    
    for l2_value, date_times in data_dict_l.items():
        print(f"L2_{l2_value}_dB_L: {date_times}")
    
    for l2_value, date_times in data_dict_r.items():
        print(f"L2_{l2_value}_dB_R: {date_times}")

# Specify the folder containing the files
folder_path = "Results/s126/"

# Print the dictionaries as command line output
print_dicts_as_command_line(folder_path)
