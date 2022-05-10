'''
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy.io import wavfile
from queue import Queue
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from scipy import signal
"""-----------------------------------------------------------------------------------------------"""
#label : on mnt/scratch/data/on/**.wav
"""-----------------------------------------------------------------------------------------------"""

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)



class wav_split(Dataset):
    def __init__(self, dataset_file_name, train_path, n_mels, noise_path, input_length, noise_prob):
        self.dataset_file_name = dataset_file_name;

        self.data_dict = {};
        self.nFiles = 0;

        self.torchfb        = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=n_mels);

        self.input_length = input_length 
        self.noise_prob = noise_prob 
        
        #noise
        augment_files   = glob.glob(os.path.join(noise_path,'*.wav'));
        self.bg_noise_audio = []
        for file in augment_files:
            noise, _ = soundfile.read(file, dtype='int16') #noise.shape = (978488,)
            self.bg_noise_audio.append(noise) #list(ndarray) -> 6개

        ### Read Training Files...
        with open(dataset_file_name) as f:
            self.lines = f.readlines()
            
        for line in self.lines: 
            
            data = line.split();                            #on on/8911b8d2_nohash_0.wav
            filename = os.path.join(train_path,data[1]);    #'/mnt/scratch/datasets/speech_commands_v0.01/on/8911b8d2_nohash_0.wav'
            keyword = data[0]                          #'on'

            if keyword not in self.data_dict:
                self.data_dict[keyword] = [filename]
            else:
                self.data_dict[keyword].append(filename)     

    def __getitem__(self, index):
        audio_batch = []
        for keyword in self.data_dict.keys():
            audio = load_wav(self.data_dict[keyword], index).astype(numpy.float) #(2, 16000)
            audio_batch.append(audio)

        #augmentation추가
        audio_augs = []
        for audio in audio_batch:
            audio_aug = []
            audio_aug.append(self.augment_wav(audio[0]))
            audio_aug.append(self.augment_wav(audio[1]))
            audio_aug = numpy.stack(audio_aug, axis=0) #(2,16000) : numpy.ndarray
            audio_augs.append(audio_aug)

        audio_aug_batch = numpy.stack(audio_augs, axis=0)

        return audio_aug_batch #audio_aug_batch.shape = (20, 2, 16000)


    def __len__(self):
        # return len(self.data_list)
        return len(self.lines)//len(self.data_dict)

    def augment_wav(self,audio):

        in_len = self.input_length
        
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            noise_start =  random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[noise_start : (noise_start + in_len)] # bg_noise.shape = (16000,)
        else:
            bg_noise = np.zeros(in_len)

        if random.random() < self.noise_prob : #silence 추가해야해
            a = random.random() * 0.1
            audio = a * bg_noise + audio

        return audio #(16000,)


def load_wav(filelist, index):

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filelist[index], dtype='int16') #audio.shape = (16000,)

    temp = filelist[:index] + filelist[index+1 :] # 중복 제거
    choice = random.choice(temp)

    audio_pos, _ = soundfile.read(choice, dtype='int16') #audio.shape = (16000,)

    ''' 패딩 추가 '''
    max_audio = 16000
  
    if len(audio) < max_audio:
        shortage = math.floor((max_audio - len(audio) + 1) / 2)
        if len(audio) % 2 == 0:
            audio = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        else :
            audio = numpy.pad(audio, (shortage, shortage-1), 'constant', constant_values=0)

    if len(audio_pos) < max_audio:
        shortage = math.floor((max_audio - len(audio_pos) + 1) / 2)
        if len(audio_pos) % 2 == 0:
            audio_pos = numpy.pad(audio_pos, (shortage, shortage), 'constant', constant_values=0)
        else :
            audio_pos = numpy.pad(audio_pos, (shortage, shortage-1), 'constant', constant_values=0)

    feats = []
    feats.append(audio)
    feats.append(audio_pos)

    feat = numpy.stack(feats, axis=0) #(2,16000) : numpy.ndarray

    return feat


def _timeshift_audio(self, data):
    shift = (16000 * self.timeshift_ms) // 1000
    shift = random.randint(-shift, shift)
    a = -min(0, shift)
    b = max(0, shift)
    data = np.pad(data, (a, b), "constant")
    return data[:len(data) - a] if a else data[b:]


def loadWAV(filename): 

    # Maximum audio length
    max_audio = 16000
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename, dtype='int16') #audio.shape = (14861,)

    audiosize = audio.shape[0]

    #Padding : audio.shape = (16000,)
    if audiosize <= max_audio:
        shortage = math.floor((max_audio - len(audio) + 1) / 2)
        if len(audio) % 2 == 0:
            audio = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0) 
        else :
            audio = numpy.pad(audio, (shortage, shortage-1), 'constant', constant_values=0)

    return audio


def get_data_loader(dataset_file_name, batch_size, nDataLoaderThread, train_path, n_mels, noise_path, input_length, noise_prob, **kwargs):

    train_dataset = wav_split(dataset_file_name, train_path, n_mels, noise_path, input_length, noise_prob)

    train_dataset[1]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nDataLoaderThread,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=1,
    )

    return train_loader