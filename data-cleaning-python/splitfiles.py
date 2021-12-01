"""
This script should be the first script used for data cleaning/loading.
Usage: python splitfiles.py path/to/tbop_coding_samples_directory
"""

import moviepy.editor as mp
import math
import os
from pydub import AudioSegment
import sys

# From DataCleaningScripts.ipynb
def split_audio_file(audiofile, outdir):
    """
    audiofile: path to file to split
    outdir: directory to write all output files
    return: nothing. This function splits audiofile and writes the 20-second files to outdir.
    """
    print('Splitting file', audiofile, 'and writing results to', outdir)
    audio_name = os.path.splitext(os.path.basename(audiofile))[0]
    t1 = 0
    t2 = 2000 # 2 seconds, in milliseconds
    my_clip = mp.VideoFileClip(audiofile)
    my_clip.audio.write_audiofile(os.path.join(outdir, audio_name + '.wav'))
    audio = AudioSegment.from_wav(os.path.join(outdir, audio_name + '.wav'))

    total_seconds = math.ceil(audio.duration_seconds)
    count = 0
    for i in range(0, total_seconds, 20):
        count += 1
        split_fn = "_"+str(i) + '-' + "_"+str(i+20) +".wav"
        t1 = i
        t2 = i + 20
        t1 *= 1000
        t2 *= 1000
        newAudio = audio[t1:t2]
        newAudio.export(os.path.join(outdir, split_fn), format='wav')
        if i == total_seconds - 20:
            print('All split successfully')

def split_all(datadir):
    """
    datadir: dir containing subdirectories 'Videos' and 'Codes'
    return: nothing. Creates new subdirectories for the split files and splits all audio files.
    """
    os.mkdir(os.path.join(datadir, 'SplitVideos'))
    for asf_file in os.listdir(os.path.join(datadir, 'Videos')):
        print('.asf file:', asf_file)
        video_name = os.path.splitext(asf_file)[0]
        os.mkdir(os.path.join(datadir, 'SplitVideos', video_name))
        split_audio_file(os.path.join(datadir, 'Videos', asf_file), os.path.join(datadir, 'SplitVideos', video_name))

if __name__ == '__main__':
    datadir = sys.argv[1]
    split_all(datadir)
