#%%
import os, glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns

#%%
path = "/mnt/work4/datasets/keyword/qualcomm_keyword_speech_dataset"
wav_files = glob.glob(path+'/*/*/*.wav', recursive=True)
wav_dict = {}
for wav_file in wav_files:
    keyword = wav_file.split('/')[-3]
    if keyword not in wav_dict:
        wav_dict[keyword] = []
    else:
        wav_dict[keyword].append(wav_file)
# %%
file2wav = {}
durations = []
max_dur = 0
max_dur_file = ''
for wav_file in wav_files:
    file2wav[wav_file] = {}
    audio, sr = sf.read(wav_file)
    if sr != 16000:
        print('sampling rate is not 16000!')
    file2wav[wav_file]['audio'] = audio
    file2wav[wav_file]['SR'] = sr
    duration = len(audio) / sr
    file2wav[wav_file]['duration'] = duration
    if duration > max_dur:
        max_dur = duration
        max_dur_file = wav_file
    durations.append(duration)
# %%
durations = np.array(durations)
durations = np.sort(durations)
sns.distplot(durations)

# %%
