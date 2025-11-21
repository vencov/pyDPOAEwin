import os
import re

def create_dicts(folder_path):
    # Define the pattern to match the date, time, and F2b in the filenames
    date_time_f2b_pattern = re.compile(r'(\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_F2b_(\d+)Hz)')

    # Create dictionaries to store unique date_time_f2b strings for each sweep rate and ear side
    sweep_rate_dict_l = {}
    sweep_rate_dict_r = {}

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if 'p4swDPOAE' in filename:
            match = date_time_f2b_pattern.search(filename)
            if match:
                date_time_f2b = match.group(1)
                sweep_rate = int(match.group(2))

                if "_L_" in filename:
                    if sweep_rate not in sweep_rate_dict_l:
                        sweep_rate_dict_l[sweep_rate] = {date_time_f2b}
                    else:
                        sweep_rate_dict_l[sweep_rate].add(date_time_f2b)
                elif "_R_" in filename:
                    if sweep_rate not in sweep_rate_dict_r:
                        sweep_rate_dict_r[sweep_rate] = {date_time_f2b}
                    else:
                        sweep_rate_dict_r[sweep_rate].add(date_time_f2b)

    # Convert sets to sorted lists and prepend the folder path
    sweep_rate_dict_l = {k: [folder_path] + sorted(list(v)) for k, v in sweep_rate_dict_l.items()}
    sweep_rate_dict_r = {k: [folder_path] + sorted(list(v)) for k, v in sweep_rate_dict_r.items()}

    return sweep_rate_dict_l, sweep_rate_dict_r

def print_dicts_as_command_line(folder_path):
    sweep_rate_dict_l, sweep_rate_dict_r = create_dicts(folder_path)
    
    for sweep_rate, date_times in sweep_rate_dict_l.items():
        print(f"Sweep Rate {sweep_rate}Hz_L: {date_times}")
    
    for sweep_rate, date_times in sweep_rate_dict_r.items():
        print(f"Sweep Rate {sweep_rate}Hz_R: {date_times}")

# Specify the folder containing the files
folder_path = "Results/s145/"

# Print the dictionaries as command line output
print_dicts_as_command_line(folder_path)
