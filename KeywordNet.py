'''
* Original Code : https://github.com/clovaai/voxceleb_trainer/blob/master/SpeakerNet.py
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

#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchaudio import transforms
import numpy, sys, random
import time, os, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV, loadSilence
from ConfModel import *

class KeywordNet(nn.Module):

    def __init__(self, model, optimizer, scheduler, trainfunc, mixedprec, fine_tuning, n_mels, **kwargs):
        super(KeywordNet, self).__init__();

        self.trainfunc = trainfunc
        KeywordNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = KeywordNetModel(**kwargs).cuda();
        ## For fine-tunning t-SNE
        # self.__S__.fc = nn.Linear(1000,20)

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).cuda();

        self.__S2E__ = ConfModelBC(num_layers=1, **kwargs).cuda();

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(list(self.__S__.parameters()) + list(self.__L__.parameters()), **kwargs);
        self.__optimizerS2E__ = Optimizer(self.__S2E__.parameters(), **kwargs);

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        assert self.lr_step in ['epoch', 'iteration']
        n_fft = 480
        # n_fft = 960
        hop_length = 160
        # hop_length = 320
        self.mfcc = transforms.MFCC(sample_rate=16000, n_mfcc = 40, melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': hop_length,'mel_scale': 'htk'}).cuda()
        self.mixedprec = mixedprec
        
        if mixedprec:
            self.scaler = GradScaler() 

        ## ===== ===== ===== ===== ===== ===== ===== =====
        ''' Fine-tunning '''
        ## ===== ===== ===== ===== ===== ===== ===== =====
        if fine_tuning:
            for name, param in self.__S__.named_parameters():
                if name in ['conv0.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'conv1.weight',
                                            'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'conv2.weight',   
                                            'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked', 'conv3.weight']:
                    param.requires_grad = False

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Train network '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, epoch, loader, alpha=1, num_steps=1):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0;    

        tstart = time.time()
        for data, labels in loader:
            self.zero_grad();

            batchsize, minibatchsize, num_wav, num_sample = data.shape
            data = data.view(-1, num_wav, num_sample) 

            data = data.cuda()
            data = data.to(torch.float32)
            data = data.transpose(0,1)
            data = self.mfcc(data).transpose(2,3)

            feat = []
            for inp in data: 
                with autocast(enabled = self.mixedprec):
                    outp = self.__S__.forward(inp)
                feat.append(outp)

            feat = torch.stack(feat, dim=1).squeeze()
            
            feat = feat.view(batchsize, minibatchsize, num_wav, -1) 
            
            batchloss = []
            batchprec = []

            with autocast(enabled = self.mixedprec):
                for _, one_feat in enumerate(feat):
                    if self.trainfunc == 'angleproto_mean':
                        _nloss, _prec1 = self.__L__.forward(one_feat,epoch,labels)
                    else:
                        _nloss, _prec1 = self.__L__.forward(one_feat)
                    batchloss.append(_nloss)
                    batchprec.append(_prec1)
                nloss = torch.mean(torch.stack(batchloss))
                prec1 = torch.mean(torch.stack(batchprec))

            loss    += nloss.detach().cpu();
            top1    += prec1
            counter += 1;
            index   += stepsize;
            
            if self.mixedprec:
                self.scaler.scale(nloss).backward();
                self.scaler.unscale_(self.__optimizer__);
                torch.nn.utils.clip_grad_norm_(self.__S__.parameters(), 5);
                self.scaler.step(self.__optimizer__);
                self.scaler.update();   
            else:
                nloss.backward();
                self.__optimizer__.step();

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index));
            sys.stdout.write("Loss %f EER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
            sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()


        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Train network (Classification) '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network_classify(self, loader):
        self.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0

        tstart = time.time()

        for data, data_label in loader:
            data = data.cuda()
            data = self.mfcc(data).transpose(1, 2)

            self.zero_grad()

            data = self.__S__.forward(data)
            label = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__L__.forward(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__L__.forward(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
            sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
            sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Evaluate accuracy from list '''
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def evaluateAccuracyFromList(self, num_shots, enrollfilename, listfilename, enroll_path='', test_path='', noise_path=''):
        
        self.eval();
        target_keys = '__silence__, him, about, out, its, your, their, will, some, the, when'.split(', ')

        files       = {}
        test_feat_by_key = {}

        feat_by_key = {}
        enroll_files = {}
        centroid_by_key = {}

        with open(enrollfilename) as enrollfile:
            while True:
                line = enrollfile.readline();
                if (not line):
                    break;

                data = line.split();
                if len(data) != 2:
                    sys.stderr.write("Too many or too little data in one line.")
                    exit()

                key = data[0]
                filename = data[1]

                if key in target_keys:
                    if key in enroll_files:
                        enroll_files[key].append(filename)
                    else:
                        enroll_files[key] = [filename]
                else:
                    if '__unknown__' in enroll_files:
                        enroll_files['__unknown__'].append(filename)
                    else:
                        enroll_files['__unknown__'] = [filename]
        for key, audios in enroll_files.items():
            feat_by_key[key] = []

            for audio in audios:
                inp = torch.FloatTensor(loadWAV(os.path.join(enroll_path, audio))).cuda()
                with torch.no_grad():
                    inp = self.mfcc(inp)
                    inp = inp.transpose(0, 1).unsqueeze(0)
                    feat = self.__S__.forward(inp).detach().cpu()
                feat_by_key[key].append(feat)

        feat_by_key['__silence__'] = []
        for i in range(num_shots):
            inp = torch.FloatTensor(loadSilence(noise_path)).cuda()
            with torch.no_grad():
                inp = self.mfcc(inp)
                inp = inp.transpose(0, 1).unsqueeze(0)
                feat = self.__S__.forward(inp).detach().cpu()
            feat_by_key['__silence__'].append(feat)

        for key, feats in feat_by_key.items():
            centroid_by_key[key] = torch.mean(torch.stack(feats), axis=0)

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): 
                    break;

                data = line.split();

                if len(data) != 2:
                    sys.stderr.write("Too many data in one line")
                    exit()

                key = data[0]
                filename = data[1]

                if key in target_keys:
                    if key in files:
                        files[key].append(filename)
                    else:
                        files[key] = [filename]
                else:
                    if '__unknown__' in files:
                        files['__unknown__'].append(filename)
                    else:
                        files['__unknown__'] = [filename]

        for key, audios in files.items():
            test_feat_by_key[key] = []
            for audio in audios:
                inp = torch.FloatTensor(loadWAV(os.path.join(test_path, audio))).cuda()
                with torch.no_grad():
                    inp = self.mfcc(inp)
                    inp = inp.transpose(0, 1).unsqueeze(0)
                    feat = self.__S__.forward(inp).detach().cpu()
                test_feat_by_key[key].append(feat)
        test_feat_by_key['__silence__'] = []

        for i in range(300) : # 300 is # of test samples per class
            inp = torch.FloatTensor(loadSilence(noise_path)).cuda()
            with torch.no_grad():
                inp = self.mfcc(inp)
                inp = inp.transpose(0, 1).unsqueeze(0)
                feat = self.__S__.forward(inp).detach().cpu()
            test_feat_by_key['__silence__'].append(feat)


        all_multi_labels = []
        all_preds = []
        all_scores = []
        all_labels = []

        del centroid_by_key['__unknown__']
        del centroid_by_key['__silence__']

        for key, feats in test_feat_by_key.items():
            for feat in feats:
                cos_sims = {}
                for _key, _ in centroid_by_key.items():
                    if True:
                        feat = F.normalize(feat, dim=1)
                        centroid_by_key[_key] = F.normalize(centroid_by_key[_key], dim=1)
                    cos_sim = F.cosine_similarity(feat.unsqueeze(-1), centroid_by_key[_key].unsqueeze(-1).transpose(0, 2))
                    cos_sims[_key] = cos_sim
                    all_scores.append(cos_sim.squeeze(0).numpy()[0])
                    if key == _key:
                        all_labels.append(1)
                    else:
                        all_labels.append(0)
                pred = max(cos_sims, key=cos_sims.get)

                if key == '__unknown__' or key == '__silence__':
                    continue;
                all_preds.append(pred)
                all_multi_labels.append(key)
        return (all_preds, all_multi_labels, all_scores, all_labels)

    def saveParameters(self, path):
        
        torch.save(self.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Load parameters '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);