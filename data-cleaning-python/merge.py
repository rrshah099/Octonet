import sys
import os
import re
import pandas as p
from pydub import AudioSegment
import numpy as np
from sklearn.linear_model import LogisticRegression as LR

import csv_combo

def find_file(filenames, begin, end):
    """
    filenames: list of filenames
    begin: first number to look for in filename
    end: second number to look for in filename
    return: the first filename that contains both begin and end, or None if it doesn't exist
    """
    filenames = sorted(filenames)
    pattern = re.compile(r'.(\d+)-.(\d+)') # TODO only works with current filename formatting
    b_str = str(begin)
    e_str = str(end)
    for name in filenames:
        m = pattern.match(name)
        if m is None:
            continue
        if b_str == m.group(1) and e_str == m.group(2):
            return name
    return None

def get_begin_end(timepoint):
    """
    timepoint: string from timepoint column
    return: pair of ints (first timepoint in seconds, second timepoint in seconds)
    """
    print(timepoint)
    times = timepoint.split('-')
    assert len(times) == 2
    begintime = times[0]
    endtime = times[1]
    begintime_split = begintime.split(':')
    assert len(begintime_split) == 2
    if begintime_split[0] == "":
        begin = int(begintime_split[1])
    else:
        begin = 60 * int(begintime_split[0]) + int(begintime_split[1])
    endtime_split = endtime.split(':')
    assert len(endtime_split) == 2
    if endtime_split[0] == "":
        end = int(endtime_split[1])
    else:
        end = 60 * int(endtime_split[0]) + int(endtime_split[1])
    return begin, end

def augment_dataframes(spreadsheet_dict, datadir, filter=True, files_to_load=None):
    """
    spreadsheet_dict: dictionary loaded by get_all_spreadsheets
    datadir: directory containing subdirectories 'Codes' and 'Videos'
    filter: boolean. If True, only include rows with Mode 3, 15, or 18.
    files_to_load: if None, this function will load the audio clips into all dataframes.
        Otherwise, this should be a list of strings, where each string is a
        key in spreadsheet_dict. Only those files will have their audio clips loaded.

    return: nothing. This function mutates spreadsheet_dict by
    augmenting each dataframe to add the 'Audio Clip' column containing
    20-second audio snippets.
    """
    if files_to_load is None:
        keys = list(spreadsheet_dict.keys())
    else:
        keys = files_to_load
    for excel_filename in keys:
        df = spreadsheet_dict[excel_filename]
        print('Now augmenting dataframe for', excel_filename)
        assert len(excel_filename) >= 8
        filename_prefix = excel_filename[:8]
        audios_dir = os.path.join(datadir, 'SplitVideos', filename_prefix)
        print('audios_dir:', audios_dir)
        audio_files = os.listdir(audios_dir)
        # Load file for each row, create new column, add that column
        audios = []
        debug = True
        for tp in df['Timepoint']:
            b, e = get_begin_end(tp)
            fname = find_file(audio_files, b, e)
            if debug:
                print('tp:', tp, '\tfname:', fname, '\tb:', b, '\te:', e)
            if fname is None:
                print('WARNING: file for "{}" is missing'.format(tp))
                audio = None
            else:
                audio = AudioSegment.from_wav(os.path.join(audios_dir, fname))
                # NOTE: if file fails to load, pydub handles it by truncating audio data
                if debug:
                    print('Length of raw bytes loaded:', len(audio.raw_data))
            audios.append(audio)
        df['Audio Clip'] = audios
        print('Number of rows:', df.shape[0])
        if filter:
            print('Filtering by mode: 3, 15, or 18')
            df_filtered = df[df['Mode'].isin([3, 15, 18])]
            print('Number of rows:', df_filtered.shape[0])
        else:
            df_filtered = df
        
        spreadsheet_dict[excel_filename] = df_filtered

        # print('Starting logistic regression for', excelfile)
        # df_nonnull = df_filtered[~df_filtered['Audio Clip'].isnull()]
        # labels = df_nonnull['Mode'].tolist()
        # labels_binary = [(1 if label == 3 else 0) for label in labels]
        # print('Binary labels (1 means mode 3, 0 means mode 15/18):', labels_binary)
        # audios_list = [np.array(a.get_array_of_samples()) for a in df_nonnull['Audio Clip']]
        # audios_array = np.stack(audios_list)
        # print('audios_array.shape:', audios_array.shape)
        # lrmodel = LR()
        # lrmodel.fit(audios_array, labels_binary)
        # print('Logistic regression score:')
        # s = lrmodel.score(audios_array, labels_binary)
        # print(s)

def load_and_augment_data(datadir, files_to_load=None):
    """
    This is the primary function to perform data loading and cleaning.

    datadir: dir containing subdirectories 'Videos' and 'Codes', as well as 'SplitVideos'
    files_to_load: see documentation in augment_dataframes. If this is None, all the data will be loaded.
    return: a spreadsheet_dict object augmented by augment_dataframes
    """
    spreadsheet_dict = csv_combo.get_all_spreadsheets(datadir)
    augment_dataframes(spreadsheet_dict, datadir, files_to_load=files_to_load)
    return spreadsheet_dict
