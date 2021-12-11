import sys
import os
import re
import pandas as p
from pydub import AudioSegment
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
from sklearn.linear_model import Lasso
from sklearn.svm import SVC

import csv_combo
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random

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


def augment_dataframes(spreadsheet_dict, datadir, filter=True):
    """
    spreadsheet_dict: dictionary loaded by get_all_spreadsheets
    datadir: directory containing subdirectories 'Codes' and 'Videos'
    filter: boolean. If True, only include rows with Mode 3, 15, or 18.

    return: nothing. This function mutates spreadsheet_dict by
    augmenting each dataframe to add the 'Audio Clip' column containing
    20-second audio snippets.
    """
    keys = list(spreadsheet_dict.keys())
    i = 0
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
        energies = []
        mfccs = []
        debug = True
        j = 0
        for tp in df['Timepoint']:
            b, e = get_begin_end(tp)
            fname = find_file(audio_files, b, e)
            if debug:
                print('tp:', tp, '\tfname:', fname, '\tb:', b, '\te:', e)
            if fname is None:
                print('WARNING: file for "{}" is missing'.format(tp))
                audio = None
            else:
                audio, sample_rate = librosa.load(os.path.join(audios_dir, fname), res_type="kaiser_fast")

                energy = librosa.feature.rms(y=audio)
                energy = energy[0]
                avg_energy = sum(energy) / len(energy)

                # plt.figure(figsize=(12,4))
                # plt.plot(audio)
                # plt.show()
                # print(df['Mode'][j])

                mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
                mfccs_processed = np.mean(mfcc.T, axis=0)
                avg_mfcc = sum(mfccs_processed) / (len(mfccs_processed))

                audio = np.stack(audio)
                energy = np.stack(energy)
                mfccs_processed = np.stack(mfccs_processed)
                # audio = AudioSegment.from_wav(os.path.join(audios_dir, fname))
                # NOTE: if file fails to load, pydub handles it by truncating audio data
                # if debug:
                # print('Length of raw bytes loaded:', len(audio.raw_data))
                j += 1
            audios.append(audio)
            energies.append(energy)
            mfccs.append(mfccs_processed)

        df['Audio Clip'] = audios
        df['Energy'] = energies
        df['Avg MFCC'] = mfccs
        print('Number of rows:', df.shape[0])
        if filter:
            print('Filtering by mode: 3, 15, or 18')
            df_filtered = df[df['Mode'].isin([3, 15, 18])]
            print('Number of rows:', df_filtered.shape[0])
        else:
            df_filtered = df

        spreadsheet_dict[excel_filename] = df_filtered
        i += 1


def load_and_augment_data(datadir):
    """
    This is the primary function to perform data loading and cleaning.

    datadir: dir containing subdirectories 'Videos' and 'Codes', as well as 'SplitVideos'
    return: a spreadsheet_dict object augmented by augment_dataframes
    """

    spreadsheet_dict = csv_combo.get_all_spreadsheets(datadir)
    print(len(spreadsheet_dict))
    augment_dataframes(spreadsheet_dict, datadir)
    return spreadsheet_dict

if __name__ =="__main__":
    df = load_and_augment_data("TBOP coding samples")

    # Combining all the dataframes for each file into one large dataframe
    i = 0
    for key in df.keys():
        if i == 0:
            one_big_df = df[key].drop(
                ['Timepoint', 'District ID', 'School ID', 'Teacher ID', 'Condition', 'Obs. Entry', 'ESL Strategy',
                 'Curriculum',
                 'Physical Group', 'Activity Structure', 'Language Content', 'Lang. of Instruction(T)',
                 'Lang. of Instruction(S)', 'Round'], axis=1)
        else:
            data = df[key].drop(
                ['Timepoint', 'District ID', 'School ID', 'Teacher ID', 'Condition', 'Obs. Entry', 'ESL Strategy',
                 'Curriculum',
                 'Physical Group', 'Activity Structure', 'Language Content', 'Lang. of Instruction(T)',
                 'Lang. of Instruction(S)', 'Round'], axis=1)
            one_big_df = p.concat([one_big_df, data])
        i += 1

    labels = []
    for key in df.keys():
        labels = labels + df[key]['Mode'].tolist()
    labels_binary = [(1 if label == 3 else 0) for label in labels]
    #print(len(labels_binary))

    print(one_big_df.head(5))

    # Extracting the features and preparing them for a ML Model
    df_nonnull = one_big_df[~one_big_df['Audio Clip'].isnull()]
    audios = df_nonnull["Audio Clip"]
    energies = df_nonnull["Energy"]
    mfccs = df_nonnull["Avg MFCC"]
    labels = df_nonnull['Mode'].tolist()
    yy = [(1 if label == 3 else 0) for label in labels]
    df_nonnull = df_nonnull.drop(['Mode'], axis=1)
    audios = [np.array(a) for a in df_nonnull['Audio Clip']]
    audios = np.array(audios)
    energies = [np.array(a) for a in df_nonnull['Energy']]
    energies = np.array(energies)
    mfccs = [np.array(a) for a in df_nonnull['Avg MFCC']]
    mfccs = np.array(mfccs)
    print(mfccs.shape)

    accuracy_scores = []
    iterations = []

    for i in range(10):
        train_X, test_X, train_Y, test_Y = train_test_split(mfccs, yy, test_size=0.1,
                                                            random_state=random.randint(0, 100))
        model = RandomForestClassifier(n_estimators=1000)
        model.fit(train_X, train_Y)
        pred_Y = model.predict(test_X)
        print(pred_Y)
        print("Accuracy: " + str(accuracy_score(test_Y, pred_Y)))
        accuracy_scores.append(accuracy_score(test_Y, pred_Y))
        iterations.append(i)

    # Graphing the results
    plt.plot(iterations, accuracy_scores)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()
    print("Average Accuracy: " + str(sum(accuracy_scores) / 10))
