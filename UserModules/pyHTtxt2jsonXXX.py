"""
Input: txt of hearing thresholds
Output: json dictionary
Autor: Katerina Polakova
Date: 22.04.2024

"""

import re


"""
    Extracts information from text file into simple dictionary, line by line
    variables Frequency, Time of the tone, Volume(intensity), Time of pressed button
"""
def parse_line(line):
    freq = re.search(r"frequency\s*=\s*(\d+)\s*Hz", line)
    time = re.search(r"time:\s*(\d+)\s*ms", line)
    volume = re.search(r"volume\s*=\s*(-?\d+)\s*dB", line)
    pressed = re.search(r"Button pressed after:\s*(\d+(\.\d+)?)", line)

    # Extracting the values
    freq_value = int(freq.group(1)) if freq else None
    time_value = int(time.group(1)) if time else None
    volume_value = int(volume.group(1)) if volume else None
    pressed_value = float(pressed.group(1)) if pressed else None #was button pressed within 2sec if so how long did it take
    # Create a dictionary for the JSON output
    data_dict = {
        "frequency": freq_value, #frequency of tone
        "time": time_value, #length of tone
        "volume": volume_value, #volume/ intensity of tone
        "button_pressed": pressed_value  # NULL value signifies button was not pressed in time
    }
    return data_dict



"""
    Transform into more sophisticated dictionary
"""
def stat_freq(parsed_data_list):
    new_data_list = [] #initial dictionary with only sorted values of volumes for each frequency
    keyList = [] #list of used frequencies
    for data_dict in parsed_data_list: #for each entry from input dictionary
        frequency_value = data_dict["frequency"]
        time = data_dict["time"]
        volume = data_dict["volume"]
        pressed= data_dict["button_pressed"]
        if frequency_value not in keyList:
            keyList.append(frequency_value)
            new_data_dict = {
            "frequency": frequency_value,
            "time": time,
            "volume": [volume],
            "button_pressed": [pressed]  }
            new_data_list.append(new_data_dict)
        
        else:
            for each in new_data_list:
                time_ok = False
                if each["frequency"] == frequency_value:
                    if each["time"] == time:
                        time_ok = True
                        inserted = False
                        for j in range(len(each["volume"])):
                            if each["volume"][j] > volume:
                                each["volume"].insert(j,volume)
                                each["button_pressed"].insert(j,pressed)
                                inserted = True
                                break
                        if inserted == False:
                            each["volume"].append(volume)
                            each["button_pressed"].append(pressed)
            if time_ok == False:
                new_data_dict = {
                "frequency": frequency_value,
                "time": time,
                "volume": [volume],
                "button_pressed": [pressed]  }
                new_data_list.append(new_data_dict)
    new_data_list2 = []
    for data_dict_2 in new_data_list:
        inner_dict_list =[]
        frequency_value2 = data_dict_2["frequency"]
        time2 = data_dict_2["time"]
        volume2 = data_dict_2["volume"]
        pressed2= data_dict_2["button_pressed"]
        new_pressed = []
        KeyList2 = []
        for i in range(len(volume2)):
            if i ==0:
                new_pressed = [pressed2[i]]
                KeyList2.append(volume2[i])

            elif volume2[i] not in KeyList2:
                new_inner_data_dict = {
            "value": volume2[i-1],
            "button_pressed": new_pressed  # NULL value signifies button was not pressed in time
            }
                inner_dict_list.append(new_inner_data_dict)
                new_pressed = [pressed2[i]]
                KeyList2.append(volume2[i])
                if i == len(volume2)-1:
                    new_inner_data_dict = {
            "value": volume2[i],
            "button_pressed": new_pressed  # NULL value signifies button was not pressed in time
            }
                    inner_dict_list.append(new_inner_data_dict)

            else:
                #idx = KeyList2.index(volume2[i]) TODO not necessary already ordered
                new_pressed.append(pressed2[i])
                if i == len(volume2)-1:
                    new_inner_data_dict = {
            "value": volume2[i],
            "button_pressed": new_pressed  # NULL value signifies button was not pressed in time
                    }
                    inner_dict_list.append(new_inner_data_dict)
        new_data_dict2 = {
            "frequency": frequency_value2,
            "time": time2,
            "volume": inner_dict_list  # NULL value signifies button was not pressed in time
            }
        new_data_list2.append(new_data_dict2)

    return new_data_list2

def txt_json_convert_HT(input_txt):
    file = open(input_txt,'r')
    Lines = file.read()
    file.close()

    parsed_data_list = []
    lines_split= Lines.split('/n')# Strips the newline character

    for line in lines_split:
        parsed_data_list.append(parse_line(line))


    new_list = stat_freq(parsed_data_list) # creating new dictionary

    #writing new dictionary into json
    
    return new_list

