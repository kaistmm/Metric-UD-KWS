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
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV, loadSilence
from ConfModel import *
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class KeywordNet(nn.Module):

    def __init__(self, model, optimizer, scheduler, trainfunc, mixedprec, fine_tunning, n_mels, **kwargs):
        super(KeywordNet, self).__init__();

        self.trainfunc = trainfunc
        KeywordNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = KeywordNetModel(**kwargs).cuda();
        ## For fine-tunning t-SNE
        # self.__S__.fc = nn.Linear(1000,20)

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(fine_tunning, **kwargs).cuda();

        self.__S2E__ = ConfModelBC(num_layers=1, **kwargs).cuda();

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(list(self.__S__.parameters()) + list(self.__L__.parameters()), **kwargs);
        self.__optimizerS2E__ = Optimizer(self.__S2E__.parameters(), **kwargs);

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
        ''' Fine-tunning '''
        ## ===== ===== ===== ===== ===== ===== ===== =====
        if fine_tunning:
            for name, param in self.__S__.named_parameters():
                if name in ['conv0.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'conv1.weight',
                                            'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'conv2.weight',   
                                            'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked', 'conv3.weight']:
                # if name in ['output.weight','output.bias']:
                    param.requires_grad = False
            # Adding layers
        #     self.__S__ = nn.Sequential(
        #         self.__S__,
        #         nn.Linear(1000,20).cuda()
        #     )

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Train network '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, epoch, loader, alpha=1, num_steps=1):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0;    # EER or accuracy

        criterion   = torch.nn.CrossEntropyLoss()
        conf_labels = torch.LongTensor([1]*stepsize+[0]*stepsize).cuda()

        tstart = time.time()
        
        for data, labels in loader: #data = torch.Size([N,20,2,16000])

            self.zero_grad();

            batchsize, minibatchsize, num_wav, num_sample = data.shape
            data = data.view(-1, num_wav, num_sample) # [N*20, 2, 16000]

            data = data.cuda()
            data = data.to(torch.float32)
            data = data.transpose(0,1) # torch.Size([2,N*20,16000])
            data = self.mfcc(data).transpose(2,3) # torch.Size([2,N*20,40,101])->[2,N*20,101,40]

            feat = []
            for inp in data:  #inp.shape = [N*20,101,40]
                with autocast(enabled = self.mixedprec):
                    outp = self.__S__.forward(inp)
                # import pdb; pdb.set_trace()
                feat.append(outp) #feat[1].shape = torch.Size([N*20, 20]) = outp

            feat = torch.stack(feat, dim=1).squeeze() #feat.shape = torch.Size([N*20, 2, 20])
            
            conf_loss   = 0
            feat = feat.view(batchsize, minibatchsize, num_wav, -1) #feat.shape = torch.Size([N, 20, 2, 20])
            
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
    ''' Fine-tune network '''
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network_classify(self, loader):
        self.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:

            # data = data.transpose(1, 0).cuda()
            data = data.cuda()
            data = self.mfcc(data).transpose(1, 2)

            # import pdb; pdb.set_trace()

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

        # import pdb; pdb.set_trace()
        ## Save all features to file
        for idx, file in enumerate(setfiles):
            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file))).cuda() #torch.Size([16000])

            with torch.no_grad():

                inp1 = self.mfcc(inp1)
                inp1 = inp1.transpose(0,1).unsqueeze(0) #torch.Size([16000])->[40,101]->[1,101,40]

                ref_feat = self.__S__.forward(inp1).detach().cpu() #torch.Size([1, 20])


            filename = '%04d.wav'%idx #'000000.wav'

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
    ''' Evaluate accuracy from list '''
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def evaluateAccuracyFromList(self, num_shots, enrollfilename, listfilename, print_interval=100, enroll_path='', test_path='', num_eval=10, noise_path=''):
        
        self.eval();
        target_keys = '__silence__, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

        files       = {}
        test_feat_by_key = {}
        tstart      = time.time()

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
        wrong = 0
        correct = 0

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

        for i in range(300) : # magic number!!-> going to modify
            inp = torch.FloatTensor(loadSilence(noise_path)).cuda()
            with torch.no_grad():
                inp = self.mfcc(inp)
                inp = inp.transpose(0, 1).unsqueeze(0)
                feat = self.__S__.forward(inp).detach().cpu()
            test_feat_by_key['__silence__'].append(feat)

        for key, feats in test_feat_by_key.items():
            for feat in feats:
                cos_sims = {}
                for _key, centroids in centroid_by_key.items():
                    if True:
                        feat = F.normalize(feat, dim=1)
                        centroid_by_key[_key] = F.normalize(centroid_by_key[_key], dim=1)
                    cos_sims[_key] = F.cosine_similarity(feat.unsqueeze(-1), centroid_by_key[_key].unsqueeze(-1).transpose(0, 2))

                pred = max(cos_sims, key=cos_sims.get)
                if pred != key:
                    wrong += 1
                else:
                    correct += 1

        accuracy = correct / (correct + wrong)
        accuracy = accuracy * 100

        return accuracy; 

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

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ''' Draw t-SNE '''
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def tsne_drawer(self, listfilename, savename, print_interval=100, test_path='', num_eval=10):
        
        self.eval();
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()
    
        ## Read all lines
        with open(listfilename) as listfile: #listfilename  = '/mnt/scratch/datasets/words_filtered/test_list.txt'
            while True:
                line = listfile.readline();  #line = '1 SHOOK/SHOOK_3157-68361-0001_37.wav SHOOK/SHOOK_8494-244431-0014_6.wav\n'
                if (not line): 
                    break;

                data = line.split();         #data = ['1', 'SHOOK/SHOOK_3157-68361-0001_37.wav', 'SHOOK/SHOOK_8494-244431-0014_6.wav']

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)
        
        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        features = []
        labels = []

        for idx, file in enumerate(setfiles):
            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file))).cuda() #torch.Size([16000])

            with torch.no_grad():

                inp1 = self.mfcc(inp1)
                inp1 = inp1.transpose(0,1).unsqueeze(0) #torch.Size([16000])->[40,101]->[1,101,40]

                ref_feat = self.__S__.forward(inp1).detach().cpu() #torch.Size([1, 1000])
                
            filename = '%04d.wav'%idx #'000000.wav'

            features.append(ref_feat)
            labels.append(file.split('/')[0])
            
            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));
        
        input_feature = torch.stack(features, dim=0).squeeze(1)

        if True:
            input_feature = F.normalize(input_feature, dim=1)

        # import pdb; pdb.set_trace()

        ## t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_ref = tsne.fit_transform(input_feature)

        df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
        df['x'] = tsne_ref[:,0]
        df['y'] = tsne_ref[:,1]
        df['Label'] = labels

        sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, size=20, hue='Label', scatter_kws={"s":200, "alpha":0.5})
        plt.title('t-SNE result', weight='bold').set_fontsize('14')
        plt.xlabel('x', weight='bold').set_fontsize('10')
        plt.ylabel('y', weight='bold').set_fontsize('10')
        plt.savefig(savename, bbox_inches='tight', pad_inches=1)

    def tsne_drawer_acc(self, num_shots, enrollfilename, listfilename, savename, print_interval=100, enroll_path='', test_path='', num_eval=10, noise_path=''):
        self.eval()
        target_keys = '__silence__, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

        files       = {}
        test_feat_by_key = {}
        tstart      = time.time()

        feat_by_key = {}
        enroll_files = {}
        centroid_by_key = {}

        # with open(enrollfilename) as enrollfile:
        #     while True:
        #         line = enrollfile.readline();
        #         if (not line):
        #             break;

        #         data = line.split();

        #         if len(data) != 2:
        #             sys.stderr.write("Too many or too little data in one line.")
        #             exit()

        #         key = data[0]
        #         filename = data[1]

        #         if key in target_keys:
        #             if key in enroll_files:
        #                 enroll_files[key].append(filename)
        #             else:
        #                 enroll_files[key] = [filename]
        #         else:
        #             if '__unknown__' in enroll_files:
        #                 enroll_files['__unknown__'].append(filename)
        #             else:
        #                 enroll_files['__unknown__'] = [filename]

        # for key, audios in enroll_files.items():
        #     feat_by_key[key] = []
        #     for audio in audios:
        #         inp = torch.FloatTensor(loadWAV(os.path.join(enroll_path, audio))).cuda()
        #         with torch.no_grad():
        #             inp = self.mfcc(inp)
        #             inp = inp.transpose(0, 1).unsqueeze(0)
        #             feat = self.__S__.forward(inp).detach().cpu()
        #         feat_by_key[key].append(feat)

        # feat_by_key['__silence__'] = []
        # for i in range(num_shots):
        #     inp = torch.FloatTensor(loadSilence()).cuda()
        #     with torch.no_grad():
        #         inp = self.mfcc(inp)
        #         inp = inp.transpose(0, 1).unsqueeze(0)
        #         feat = self.__S__.forward(inp).detach().cpu()
        #     feat_by_key['__silence__'].append(feat)

        # for key, feats in feat_by_key.items():
        #     centroid_by_key[key] = torch.mean(torch.stack(feats), axis=0)

        # centroids = []
        # cent_labels = []
        # for key, feats in centroid_by_key.items():
        #     centroids.extend(feats)
        #     for i in range(len(feats)):
        #         cent_labels.append(key)

        # cent_feature = torch.stack(centroids, dim=0).squeeze(1)
        # cent_labels = [item.replace("_", "") for item in cent_labels]

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
            # import pdb; pdb.set_trace()
        test_feat_by_key['__silence__'] = []

        for i in range(300) : # magic number!!-> going to modify
            inp = torch.FloatTensor(loadSilence(noise_path)).cuda()
            with torch.no_grad():
                inp = self.mfcc(inp)
                inp = inp.transpose(0, 1).unsqueeze(0)
                feat = self.__S__.forward(inp).detach().cpu()
            test_feat_by_key['__silence__'].append(feat)

        features = []
        labels = []
        for key, feats in test_feat_by_key.items():
            features.extend(feats)
            for i in range(len(feats)):
                labels.append(key)

        feature = torch.stack(features, dim=0).squeeze(1)
        if False:
            feature = F.normalize(feature, dim=1)
        # import pdb; pdb.set_trace()
        labels = [item.replace("_", "") for item in labels]

        # import pdb; pdb.set_trace()

        ## t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        # tsne_ref = tsne.fit_transform(cent_feature)
        tsne_feats = tsne.fit_transform(feature)

        # df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
        # df['x'] = tsne_ref[:,0]
        # df['y'] = tsne_ref[:,1]
        # df['Label'] = cent_labels

        df2 = pd.DataFrame(tsne_feats, index=tsne_feats[0:,1])
        df2['x'] = tsne_feats[:,0]
        df2['y'] = tsne_feats[:,1]
        df2['Label'] = labels

        # sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, size=20, hue='Label', scatter_kws={"s":200, "alpha":0.5})
        sns.lmplot(x="x", y="y", data=df2, fit_reg=False, legend=True, size=20, hue='Label', scatter_kws={"s":100, "alpha":1})
        plt.title('t-SNE result', weight='bold').set_fontsize('14')
        plt.xlabel('x', weight='bold').set_fontsize('10')
        plt.ylabel('y', weight='bold').set_fontsize('10')
        plt.savefig(savename, bbox_inches='tight', pad_inches=1)
