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
import numpy
import random
import os
import math
import glob
import soundfile
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from scipy import signal

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

class wav_split(Dataset):
    def __init__(self, dataset_file_name, train_path, metric_batch_size, augment, fine_tuning, no_silence, musan_path, n_mels, alpha, input_length):
        self.dataset_file_name = dataset_file_name;

        self.data_dict = {};
        self.nFiles = 0;

        self.torchfb        = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=n_mels);

        self.input_length = input_length
        
        ## Noise
        self.AUG = AugmentWAV(musan_path=musan_path) 
        self.musan_path = musan_path
        self.augment    = augment
        self.fine_tuning = fine_tuning
        self.no_silence = no_silence
        self.metric_batch_size = metric_batch_size
        self.alpha  = alpha

        ### Read Training Files...
        with open(dataset_file_name) as f:
            self.lines = f.readlines()
            
        for line in self.lines:
            keyword, filename = line.split();
            filename = os.path.join(train_path, filename);

            if keyword not in self.data_dict:
                self.data_dict[keyword] = [filename]      
            else:
                self.data_dict[keyword].append(filename)

        if not self.no_silence:
            self.data_dict['__silence__'] = [] 

        self.label_to_idx = {}
        label_set = self.data_dict.keys()

        for idx, key in enumerate(label_set):
            self.label_to_idx[key] = idx

        self.sampled_data = None
        self.shuffle_dict()
        assert self.sampled_data is not None


    def __getitem__(self, index):
        audio_batch = []

        keywords, indexes = self.sampled_data[index]

        for keyword, idx in zip(keywords, indexes):
            if keyword == '__silence__':
                audio = load_silence()
            else:
                audio = load_wav(self.data_dict[keyword], idx)
            audio_batch.append(audio)

        ## Noise augmentation
        audio_augs = []
        for audio in audio_batch:
            audio_aug = []
            audio_aug.append(self.augment_wav(audio[0]))
            audio_aug.append(self.augment_wav(audio[1]))

            audio_aug = numpy.stack(audio_aug, axis=0)
            audio_augs.append(audio_aug)

        audio_aug_batch = numpy.stack(audio_augs, axis=0)

        label = numpy.stack([self.label_to_idx[k] for k in keywords], axis=0)

        return torch.FloatTensor(audio_aug_batch), torch.LongTensor(label)

    def __len__(self):
        return len(self.lines)//self.metric_batch_size

    def shuffle_dict(self):
        self.sampled_data = []
        for _ in range(self.__len__()):
            selected_dict = random.sample(list(self.data_dict.keys()), self.metric_batch_size)
            selected_index = random.sample(list(range(len(self.lines)//(len(self.data_dict)-1))), self.metric_batch_size)
            self.sampled_data.append((selected_dict, selected_index))


    def augment_wav(self,audio):
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

        return audio

#############################################
''' Noise augmentation '''
#############################################
class AugmentWAV(object):

    def __init__(self, musan_path):

        self.max_audio = 16000
        # self.max_audio = 32000

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[1,2],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

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

def gen_echo(ref, rir, filterGain): 
    rir  = numpy.multiply(rir, pow(10, 0.1 * filterGain))
    echo = signal.convolve(ref, rir, mode='full')[:len(ref)]

    return echo

def load_silence(max_audio=16000):
    audio = numpy.zeros(max_audio)
    audio_pos = numpy.zeros(max_audio)

    if audio.shape[0] != 16000 or audio_pos.shape[0] != 16000:
        print('wrong')
    # if audio.shape[0] != 32000 or audio_pos.shape[0] != 32000:
        # print('wrong')

    feats = []
    feats.append(audio)
    feats.append(audio_pos)
    feat = numpy.stack(feats, axis=0)

    return feat

def load_wav(filelist, index):
    # Read wav file and convert to torch tensor
    audio, sr = soundfile.read(filelist[index])

    temp = filelist[:index] + filelist[index+1 :]
    choice = random.choice(temp)

    audio_pos, _ = soundfile.read(choice)

    max_audio = sr * 1 # 1 second
    # max_audio = sr * 2 # 2 second
    
    len_audio = audio.shape[0]
    len_audio_pos = audio_pos.shape[0]

    #audio padding
    if len_audio < max_audio:
        shortage = math.floor((max_audio - len_audio + 1) / 2)
        if len_audio % 2 == 0:
            audio = numpy.pad(audio, (shortage,shortage), 'constant')           
        else :
            audio = numpy.pad(audio, (shortage,shortage-1), 'constant')
    else:
        margin = len_audio - 16000
        audio = audio[int(margin/2):16000 + int(margin/2)]
        # margin = len_audio - 32000
        # audio = audio[int(margin/2):32000 + int(margin/2)]

    if len_audio_pos < max_audio:
        shortage = math.floor((max_audio - len_audio_pos + 1) / 2)
        if len_audio_pos % 2 == 0:
            audio_pos = numpy.pad(audio_pos, (shortage,shortage), 'constant')           
        else:
            audio_pos = numpy.pad(audio_pos, (shortage,shortage-1), 'constant')
    else:
        margin = len_audio_pos - 16000
        audio_pos = audio_pos[int(margin/2):16000 + int(margin/2)]
        # margin = len_audio_pos - 32000
        # audio_pos = audio_pos[int(margin/2):32000 + int(margin/2)]
   
    if audio.shape[0] != 16000 or audio_pos.shape[0] != 16000 : 
        print('wrong')
    # if audio.shape[0] != 32000 or audio_pos.shape[0] != 32000 : 
        # print('wrong')

    feats = []
    feats.append(audio)
    feats.append(audio_pos)
    feat = numpy.stack(feats, axis=0)

    return feat

def loadSilence(noise_path, max_audio=16000):
    ## randomly add noise offered by GSC
    noise_files = glob.glob(os.path.join(noise_path, '*.wav'))
    noise_file = random.choice(noise_files)
    noise, _ = soundfile.read(noise_file)
    noise_start = random.randint(0, len(noise) - max_audio - 1)
    bg_noise = noise[noise_start : (noise_start + max_audio)]

    a = random.random() * 0.1
    bg_noise = bg_noise * a

    return bg_noise

def loadWAV(filename):             
    # Read wav file and convert to torch tensor
    audio, sr = soundfile.read(filename)
    len_audio = audio.shape[0]

    max_audio = sr * 1

    if len_audio < max_audio:
        shortage = math.floor((max_audio - len_audio + 1) / 2)
        if len_audio % 2 == 0:
            audio = numpy.pad(audio, (shortage,shortage), 'constant')           
        else :
            audio = numpy.pad(audio, (shortage,shortage-1), 'constant')
    else:
        margin = len_audio - 16000
        audio = audio[int(margin/2):16000 + int(margin/2)]
        # margin = len_audio - 32000
        # audio = audio[int(margin/2):32000 + int(margin/2)]

    return audio


def get_data_loader(dataset_file_name, batch_size, metric_batch_size, nDataLoaderThread, augment, fine_tuning, no_silence, musan_path, train_path, alpha, n_mels, input_length, **kwargs):

    train_dataset = wav_split(dataset_file_name, train_path, metric_batch_size, augment, fine_tuning, no_silence, musan_path, n_mels, alpha, input_length)

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