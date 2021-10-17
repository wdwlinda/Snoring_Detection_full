import os
import csv
import numpy as np
import logging

# TODO: recursively get the files should be a option
# TODO: input soring function, condition
# TODO: think about the design of shuffling and sorting, in-function or separated
def get_files(path, keys=[], is_fullpath=True, sort=True):
    """Get all the file name under the given path with assigned keys"""
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated key
    keys = list(set(keys))

    def func(root, f, file_list, is_fullpath):
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        func(root, f, file_list, is_fullpath)
            else:
                func(root, f, file_list, is_fullpath)

    if file_list:
        if sort: file_list.sort(key=len)
    else:
        if keys: 
            logging.warning(f'No file exist with key {keys}.') 
        else: 
            logging.warning(f'No file exist.') 
            
    return file_list


def single_sample_recording(path):
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        # timestamps for single sample
        timestamps = [[r[0], r[2]] for r in rows if len(r) > 0 and r[0].isdigit()]
        # TODO: file name
        # a = list(iter(rows))
        # timestamps.insert(0, list(iter(rows))[0])
    return timestamps


# TODO: offset, duration
def generate_recording_table(peak_timestamps_path, peak_infos, save_path):
    tags = ['#', 'name', '#', 'peak timestamp', 'offset', 'duration', 'start', 'end', 'label']
    content = [tags]
    with open(peak_infos, 'r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        offset_and_duration = next(rows)

    # TODO: create csv if not exist
    # TODO: if already exist? (how to judge the file should be updated?)
    # if not os.path.isfile('record_tabel.csv'):
    #     with open('record_tabel.csv', 'w', newline='') as csvfile:
    #         csv.write('')

    file_list = get_files(peak_timestamps_path, 'csv')
    count = 0
    for i, f in enumerate(file_list):
        print(f'({i+1}/{len(file_list)}) Processing {os.path.basename(f)}')
        timestamps = single_sample_recording(f)
        # timestamps.insert(0, i+1)
        for j, t in enumerate(timestamps):
            count += 1
            t.insert(0, os.path.basename(f).replace('.csv', ''))
            t.insert(0, count)
            t.extend(offset_and_duration)
            start = float(t[3])+float(t[4])
            end = start + float(t[5])
            t.extend([start, end])
        content.extend(timestamps)
    
    # Write recording table
    with open('record_tabel.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(content)



if __name__ == '__main__':
    path = rf'C:\Users\test\Downloads\snoring_test\timestamps\1630345236867_6_peak.csv'
    path = rf'C:\Users\test\Downloads\1007\env_sounds\x'
    peak_infos = rf'C:\Users\test\Downloads\peak_infos.csv'
    generate_recording_table(path, peak_infos, '')

    # path = rf'C:\Users\test\Downloads\1007\env_sounds\x'
    # files = get_files(path, 'pms')
    # for f in files:
    #     print(f)
    # x = list(set(get_files(path))-set(get_files(path, ['m4a', 'm4a'])))
    # print(60*"=")
    # for f in x:
    #     print(f)


        
    pass