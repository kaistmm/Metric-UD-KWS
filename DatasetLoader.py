'''
* Original Code : https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
* modified by jjm & Youkyum

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
import torchaudio
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

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

class wav_split(Dataset):
    def __init__(self, dataset_file_name, train_path, n_mels, noise_path, alpha, input_length, noise_prob):
        self.dataset_file_name = dataset_file_name;

        self.data_dict = {};
        self.nFiles = 0;

        self.torchfb        = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=n_mels);

        self.input_length = input_length 
        self.noise_prob = noise_prob 
        self.alpha  = alpha
        
        #noise
        augment_files   = glob.glob(os.path.join(noise_path,'*.wav'));
        self.bg_noise_audio = []
        for file in augment_files:
            noise, _ = torchaudio.load(file)
            self.bg_noise_audio.append(noise) #list(ndarray) -> 6개

        ### Read Training Files...
        with open(dataset_file_name) as f:
            self.lines = f.readlines()
            
        for line in self.lines: 

            data = line.split();                            #on on/8911b8d2_nohash_0.wav
            filename = os.path.join(train_path,data[1]);    #'/mnt/scratch/datasets/speech_commands_v0.01/on/8911b8d2_nohash_0.wav'
            keyword = data[0]            
                               #'on'
            if keyword not in self.data_dict:
                self.data_dict[keyword] = [filename]        #{'HOWS': ["/mnt/scratch/datasets/words_filtered/HOWS/HOW'S_8023-286253-0007_19.wav"]}
            else:
                self.data_dict[keyword].append(filename)     

    def __getitem__(self, index):
        audio_batch = []
        
        selected_dict = random.sample(list(self.data_dict.keys()), 320) #len(self.data_dict.keys()) = 1000 : HAD , YOU ...
        for keyword in selected_dict:                                   #len(selected_dict) = 160
            audio = load_wav(self.data_dict[keyword], index)#(2, 16000) #len(self.data_dict[keyword]) = 1000
            audio_batch.append(audio)

        #augmentation추가
        audio_augs = []
        for audio in audio_batch:
            audio_aug = []
            audio_aug.append(self.augment_wav(audio[0]))
            audio_aug.append(self.augment_wav(audio[1]))

            audio_aug = torch.stack(audio_aug, dim=0)
            audio_augs.append(audio_aug)

        audio_aug_batch = torch.stack(audio_augs, dim=0)
        return audio_aug_batch #audio_aug_batch.shape = (20, 2, 16000)


    def __len__(self):
        return len(self.lines)//len(self.data_dict)

    def augment_wav(self,audio):

        in_len = self.input_length

        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio).squeeze(0)
            noise_start =  random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[noise_start : (noise_start + in_len)] # bg_noise.shape = (16000,)
        else:
            bg_noise = torch.zeros(in_len)

        if random.random() < self.noise_prob : #silence 추가해야해
            a = random.random() * 0.1
            audio = a * bg_noise + audio 

        return audio #(16000,)


def load_wav(filelist, index):
    # Read wav file and convert to torch tensor
    audio, sample_rate = torchaudio.load(filelist[index])

    temp = filelist[:index] + filelist[index+1 :] # 중복 제거
    choice = random.choice(temp)

    audio_pos, sample_rate = torchaudio.load(choice)

    ''' 오디오 데이터 길이 맞추기 '''
    max_audio = 16000

    audio = audio.squeeze(0)
    audio_pos = audio_pos.squeeze(0)
    
    len_audio = audio.shape[0]
    len_audio_pos = audio_pos.shape[0]

    #audio padding
    if len_audio < max_audio:
        shortage = math.floor((max_audio - len_audio + 1) / 2)

        if len_audio % 2 == 0:
            m = nn.ConstantPad1d((shortage,shortage), 0)
            audio = m(audio)
            
        else :
            m = nn.ConstantPad1d((shortage, shortage-1), 0)
            audio = m(audio)

    else:
        margin = len_audio - 16000

        audio = audio[int(margin/2):16000 + int(margin/2)]

    #audio_pos padding
    if len_audio_pos < max_audio:
        shortage = math.floor((max_audio - len_audio_pos + 1) / 2)

        if len_audio_pos % 2 == 0:
            m = nn.ConstantPad1d((shortage,shortage), 0)
            audio_pos = m(audio_pos)
            
        else :
            m = nn.ConstantPad1d((shortage, shortage-1), 0)
            audio_pos = m(audio_pos)

    else:
        margin = len_audio_pos - 16000

        audio_pos = audio_pos[int(margin/2):16000 + int(margin/2)]
    
    if audio.shape[0] != 16000 or audio_pos.shape[0] != 16000 : 
        print('wrong')

    feats = []
    feats.append(audio)
    feats.append(audio_pos)

    feat = torch.stack(feats, dim=0)

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
    audio, sample_rate = torchaudio.load(filename)

    audio = audio.squeeze(0)
    len_audio = audio.shape[0]

    if len_audio < max_audio:
        shortage = math.floor((max_audio - len_audio + 1) / 2)

        if len_audio % 2 == 0:
            m = nn.ConstantPad1d((shortage,shortage), 0)
            audio = m(audio)
            
        else :
            m = nn.ConstantPad1d((shortage, shortage-1), 0)
            audio = m(audio)

    else:
        margin = len_audio - 16000

        audio = audio[int(margin/2):16000 + int(margin/2)]

    return audio


def get_data_loader(dataset_file_name, batch_size, nDataLoaderThread, train_path, alpha, n_mels, noise_path, input_length, noise_prob, **kwargs):

    train_dataset = wav_split(dataset_file_name, train_path, n_mels, noise_path, alpha, input_length, noise_prob)

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