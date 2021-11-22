import sys
import os
import re
import pandas as p
from pydub import AudioSegment


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

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Expected two paths as cmd-line args: excelfile, audios_dir')
    excelfile = sys.argv[1]
    audios_dir = sys.argv[2]
    audio_files = os.listdir(audios_dir)
    df = p.read_excel(excelfile)

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
    print('Filtering by mode: 3, 15, or 18')
    df_filtered = df[df['Mode'].isin([3, 15, 18])]
    print('Number of rows:', df_filtered.shape[0])

