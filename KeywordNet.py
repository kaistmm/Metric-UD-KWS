#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchaudio import transforms
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV
import librosa

class KeywordNet(nn.Module):

    def __init__(self, model, optimizer, scheduler, trainfunc, mixedprec, n_mels, **kwargs):
        super(KeywordNet, self).__init__();

        KeywordNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = KeywordNetModel(**kwargs).cuda();

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).cuda();

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(list(self.__S__.parameters()) + list(self.__L__.parameters()), **kwargs);

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        assert self.lr_step in ['epoch', 'iteration']
        n_fft = 512
        win_length = 400
        hop_length = 160
        self.mfcc = transforms.MFCC(sample_rate=16000, n_mfcc = 40, melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': hop_length,'mel_scale': 'htk'}).cuda()
        self.mixedprec = mixedprec
        
        if mixedprec:
            self.scaler = GradScaler() 


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Train network '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0;    # EER or accuracy

        criterion   = torch.nn.CrossEntropyLoss()

        tstart = time.time()
        
        for data in loader: #data = torch.Size([N,20,2,16000])

            self.zero_grad();
            batchsize, minibatchsize, num_wav, num_sample = data.shape
            data = data.view(-1, num_wav, num_sample) # [N*20, 2, 16000]

            data = data.cuda()
            data = data.to(torch.float32)
            data = data.transpose(0,1) #torch.Size([2,N*20,16000])
            data = self.mfcc(data).transpose(2,3) #torch.Size([2,N*20,40,101])->[2,N*20,101,40]

            feat = []
            for inp in data:  #inp.shape = [N*20,101,40]
                with autocast(enabled = self.mixedprec):
                    outp = self.__S__.forward(inp)
                feat.append(outp) #feat[1].shape = torch.Size([N*20, 20]) = outp

            feat = torch.stack(feat, dim=1).squeeze() #feat.shape = torch.Size([N*20, 2, 20])

            conf_loss   = 0
            feat = feat.view(batchsize, minibatchsize, num_wav, minibatchsize) #feat.shape = torch.Size([N, 20, 2, 20])
            
            batchloss = []
            batchprec = []

            with autocast(enabled = self.mixedprec):
                for _, one_feat in enumerate(feat):
                    _nloss, _prec1 = self.__L__.forward(one_feat,None)
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
    ''' Evaluate from list '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, listfilename, print_interval=100, test_path='', num_eval=10):
        
        self.eval();
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): 
                    break;

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)
        
        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file))).cuda() #torch.Size([16000])

            with torch.no_grad():

                inp1 = self.mfcc(inp1)
                inp1 = inp1.transpose(0,1).unsqueeze(0) #torch.Size([16000])->[40,101]->[1,101,40]

                ref_feat = self.__S__.forward(inp1).detach().cpu() #torch.Size([1, 20])

            filename = '%06d.wav'%idx #'000000.wav'

            feats[file]     = ref_feat 
            # feats = {'bed/004ae714_nohash_1.wav': tensor([[ 2.6805, -3.2256, ... ,-0.2719]])} : feature 저장

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines): 

            data = line.split(); # data : ['1', 'bed/df1d5024_nohash_0.wav', 'bed/370844f7_nohash_0.wav']

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1) #torch.Size([1, 20])
                com_feat = F.normalize(com_feat, p=2, dim=1) #torch.Size([1, 20])

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #dist.shape = (1,20)
            score = -1 * numpy.mean(dist); #-0.012503666803240776

            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2]) #['bed/df1d5024_nohash_0.wav bed/370844f7_nohash_0.wav']

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')

        return (all_scores, all_labels, all_trials);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Save parameters '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

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