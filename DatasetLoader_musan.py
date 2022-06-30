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
    def __init__(self, dataset_file_name, train_path, dict_size, augment, musan_path, rir_path, n_mels, alpha, input_length):
        self.dataset_file_name = dataset_file_name;

        self.data_dict = {};
        self.nFiles = 0;

        self.torchfb        = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=n_mels);

        self.input_length = input_length
        
        ## Noise
        self.AUG = AugmentWAV(musan_path=musan_path, rir_path=rir_path) 
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment
        self.dict_size = dict_size
        self.alpha  = alpha

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
        selected_dict = random.sample(list(self.data_dict.keys()), self.dict_size) #len(self.data_dict.keys()) = 1000 : HAD , YOU ...
        for keyword in selected_dict:                                   #len(selected_dict) = 160
            audio = load_wav(self.data_dict[keyword], index)#(2, 16000) #len(self.data_dict[keyword]) = 1000
            audio_batch.append(audio)

        ## Noise augmentation
        audio_augs = []
        for audio in audio_batch:
            audio_aug = []
            audio_aug.append(self.augment_wav(audio[0]))
            audio_aug.append(self.augment_wav(audio[1]))

            # audio_aug = torch.stack(audio_aug, dim=0)
            audio_aug = numpy.stack(audio_aug, axis=0)
            audio_augs.append(audio_aug)

        # audio_aug_batch = torch.stack(audio_augs, dim=0)
        audio_aug_batch = numpy.stack(audio_augs, axis=0)
        return torch.FloatTensor(audio_aug_batch) #audio_aug_batch.shape = (20, 2, 16000)


    def __len__(self):
        return len(self.lines)//len(self.data_dict)

    def augment_wav(self,audio):

        # in_len = self.input_length
        if self.augment:
            augtype = random.randint(0,4)
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

        # self.max_frames = max_frames
        # self.max_audio  = max_audio = max_frames * 160 + 240
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

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):
        # audio = audio.numpy()
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise)
            # noiseaudio = noiseaudio.numpy()
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noiseaudio = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            noises.append(numpy.expand_dims(noiseaudio, axis=1))
        
        noise_audio = numpy.concatenate(noises,axis=1)   
        audio = numpy.sum(noise_audio, axis=1, keepdims=True).squeeze(1) + audio
        # audio = torch.from_numpy(audio)
        return audio

    def reverberate(self, audio):
        # audio = audio.numpy()
        rir_file    = random.choice(self.rir_files)
        
        rir     = loadWAV(rir_file)
        # rir     = rir.numpy()
        rir     = numpy.expand_dims(rir.astype(numpy.float),0)
        audio   = numpy.expand_dims(audio.astype(numpy.float),0)
        rir     = rir / numpy.sqrt(numpy.sum(rir**2))

        audio = signal.convolve(audio, rir, mode='full')[:,:self.max_audio]
        audio = audio.squeeze(0)
        return  audio


def load_wav(filelist, index):
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filelist[index])

    temp = filelist[:index] + filelist[index+1 :] # 중복 제거
    choice = random.choice(temp)

    audio_pos, sample_rate = soundfile.read(choice)

    ''' 오디오 데이터 길이 맞추기 '''
    max_audio = 16000
    # audio = audio.squeeze(0)
    # audio_pos = audio_pos.squeeze(0)
    
    len_audio = audio.shape[0]
    len_audio_pos = audio_pos.shape[0]

    #audio padding
    if len_audio < max_audio:
        shortage = math.floor((max_audio - len_audio + 1) / 2)

        if len_audio % 2 == 0:
            audio = numpy.pad(audio, (shortage,shortage), 'constant')
            
        else :
            audio = numpy.pad(audio, (shortage,shortage-1), 'constant')
            # m = nn.ConstantPad1d((shortage, shortage-1), 0)
            # audio = m(audio)

    else:
        margin = len_audio - 16000

        audio = audio[int(margin/2):16000 + int(margin/2)]

    #audio_pos padding
    if len_audio_pos < max_audio:
        shortage = math.floor((max_audio - len_audio_pos + 1) / 2)

        if len_audio_pos % 2 == 0:
            audio_pos = numpy.pad(audio_pos, (shortage,shortage), 'constant')
            # m = nn.ConstantPad1d((shortage,shortage), 0)
            # audio_pos = m(audio_pos)
            
        else :
            audio_pos = numpy.pad(audio_pos, (shortage,shortage-1), 'constant')
            # m = nn.ConstantPad1d((shortage, shortage-1), 0)
            # audio_pos = m(audio_pos)

    else:
        margin = len_audio_pos - 16000

        audio_pos = audio_pos[int(margin/2):16000 + int(margin/2)]
    
    if audio.shape[0] != 16000 or audio_pos.shape[0] != 16000 : 
        print('wrong')

    feats = []
    # audio = torch.from_numpy(audio)
    # audio_pos = torch.from_numpy(audio_pos)
    feats.append(audio)
    feats.append(audio_pos)
    feat = numpy.stack(feats, axis=0)
    # feat = torch.stack(feats, dim=0)

    # return torch.FloatTensor(feat)
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


def get_data_loader(dataset_file_name, batch_size, dict_size, nDataLoaderThread, augment, musan_path, rir_path, train_path, alpha, n_mels, input_length, **kwargs):

    train_dataset = wav_split(dataset_file_name, train_path, dict_size, augment, musan_path, rir_path, n_mels, alpha, input_length)

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