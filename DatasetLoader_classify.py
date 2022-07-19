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

class wav_loader(Dataset):
    def __init__(self, dataset_file_name, train_path, sample_per_class, augment, musan_path, rir_path, n_mels, input_length):
        self.dataset_file_name = dataset_file_name;
        self.sample_per_class = sample_per_class
        self.input_length = input_length
        self.data_list = []
        self.data_label = []
        
        ## Noise
        self.AUG = AugmentWAV(musan_path=musan_path, rir_path=rir_path) 
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment        
        ### Read Training Files...
        with open(self.dataset_file_name) as f:
            self.lines = f.readlines()

        for line in self.lines:
            data = line.split()
            filename = os.path.join(train_path, data[1])
            keyword = data[0]

            self.data_list.append(filename)
            self.data_label.append(keyword)

        for i in range(self.sample_per_class):
            self.data_label.append('__silence__')

        self.label_to_idx = {}
        label_set = set(self.data_label)

        for idx, key in enumerate(label_set):
            self.label_to_idx[key] = idx

    def __getitem__(self, index):
        if self.data_label[index] == '__silence__':
            audio = loadSilence()
        else:
            audio = loadWAV(self.data_list[index])
        
        audio = self.augment_wav(audio)
                
        return torch.FloatTensor(audio), self.label_to_idx[self.data_label[index]]

    def __len__(self):
        return len(self.data_list)

    def augment_wav(self, audio):

        # in_len = self.input_length
        if self.augment:
            augtype = random.randint(0,5)
            if augtype == 1:
                audio   = self.AUG.reverberate(audio)
            elif augtype == 2:
                audio   = self.AUG.additive_noise('music',audio)
            elif augtype == 3:
                audio   = self.AUG.additive_noise('speech',audio)
            elif augtype == 4:
                audio   = self.AUG.additive_noise('noise',audio)

        return audio #(16000,)

#############################################
''' Noise augmentation '''
#############################################
class AugmentWAV(object):

    def __init__(self, musan_path, rir_path):

        self.max_audio = 16000

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[1,2],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        # self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));
        self.rir = numpy.load('rir.npy')


    def additive_noise(self, noisecat, audio):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise)

            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noiseaudio = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            noises.append(numpy.expand_dims(noiseaudio, axis=1))
        
        noise_audio = numpy.concatenate(noises,axis=1)   
        audio = numpy.sum(noise_audio, axis=1, keepdims=True).squeeze(1) + audio

        return audio

    def reverberate(self, audio):
        SIGPRO_MIN_RANDGAIN = -7
        SIGPRO_MAX_RANDGAIN = 3

        rir_filts = random.choice(self.rir)
        rir_gains = numpy.random.uniform(SIGPRO_MIN_RANDGAIN, SIGPRO_MAX_RANDGAIN, 1)

        audio = gen_echo(audio, rir_filts, rir_gains)

        return  audio

def gen_echo(ref, rir, filterGain): ## for RIR
    rir  = numpy.multiply(rir, pow(10, 0.1 * filterGain))
    echo = signal.convolve(ref, rir, mode='full')[:len(ref)]

    return echo

def loadSilence(noise_path, max_audio=16000):
    audio = numpy.zeros(max_audio)

    return audio

def loadWAV(filename):             
    # Maximum audio length
    max_audio = 16000

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    # audio = audio.squeeze(0)
    len_audio = audio.shape[0]

    if len_audio < max_audio:
        shortage = math.floor((max_audio - len_audio + 1) / 2)

        if len_audio % 2 == 0:
            audio = numpy.pad(audio, (shortage,shortage), 'constant')
            # m = nn.ConstantPad1d((shortage,shortage), 0)
            # audio = m(audio)
            
        else :
            audio = numpy.pad(audio, (shortage,shortage-1), 'constant')
            # m = nn.ConstantPad1d((shortage, shortage-1), 0)
            # audio = m(audio)

    else:
        margin = len_audio - 16000

        audio = audio[int(margin/2):16000 + int(margin/2)]

        ## Random Start
        # noise_start =  random.randint(0, len_audio - max_audio - 1)
        # audio = audio[noise_start : (noise_start + max_audio)] # bg_noise.shape = (16000,)

    return audio


def get_data_loader_classify(dataset_file_name, batch_size, nDataLoaderThread, train_path, sample_per_class, augment, musan_path, rir_path, n_mels, input_length, **kwargs):

    train_dataset = wav_loader(dataset_file_name, train_path, sample_per_class, augment, musan_path, rir_path, n_mels, input_length)

    # train_dataset[1]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nDataLoaderThread,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=1,
    )

    return train_loader