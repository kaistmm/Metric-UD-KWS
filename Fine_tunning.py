'''
* Original Code : https://github.com/clovaai/voxceleb_trainer/blob/master/trainSpeakerNet.py
* modified by jjm

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

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
import zipfile
import datetime
from tuneThreshold import *
from KeywordNet_fine import KeywordNet
from DatasetLoader import get_data_loader

parser = argparse.ArgumentParser(description = "KeywordNet");

parser.add_argument('--config', type=str, default=None,  help='Config YAML file');

## Data loader
parser.add_argument('--batch_size', type=int, default=20,  help='Batch size');
parser.add_argument('--nDataLoaderThread', type=int, default=20, help='Number of loader threads');

## Training details
parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=150, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="angleproto",    help='Loss function');
parser.add_argument('--mixedprec',      dest='mixedprec', type=bool,  default=False, help='Enable mixed precision training');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Load and save
parser.add_argument('--initial_model',  type=str, default="/home/jjm/ex/keyword/KWS_v1/AIRSpeech_UserDefinedWWD/data/exp01_1000_320/model/model000000071.model", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="", help='Path for model and logs');

# Training and test data
# parser.add_argument('--train_list',     type=str,   default="/mnt/scratch/datasets/words_filtered/train_list.txt",     help='Train list');
# parser.add_argument('--test_list',      type=str,   default="/mnt/scratch/datasets/words_filtered/test_list.txt",     help='Evaluation list');
# parser.add_argument('--train_path',     type=str,   default="/mnt/scratch/datasets/words_filtered", help='Absolute path to the train set');
# parser.add_argument('--test_path',      type=str,   default="/mnt/scratch/datasets/words_filtered", help='Absolute path to the test set');
parser.add_argument('--noise_path',     type=str,   default="/mnt/scratch/datasets/speech_commands_v0.01/_background_noise_", help='Absolute path to the test set');

#Google speech dataset
parser.add_argument('--train_list',     type=str,   default="/mnt/scratch/datasets/speech_commands_v0.01/train_list.txt",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="/mnt/scratch/datasets/speech_commands_v0.01/test_list.txt",     help='Evaluation list');
parser.add_argument('--train_path',     type=str,   default="/mnt/scratch/datasets/speech_commands_v0.01", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="/mnt/scratch/datasets/speech_commands_v0.01", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--model',          type=str,   default="ResNet15",     help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=1000,    help='Embedding size in the last FC layer (the number of classes at training');
# parser.add_argument('--use_dilation',   type=bool,   default=True,    help='Dilation');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

## Edited
parser.add_argument('--input_length',      type=int,  default=16000,  help='input length(default=16000)')
parser.add_argument('--noise_prob',      type=float,  default=0.8,  help='noise prob')

## Specific to environment removal
parser.add_argument('--alpha', type=float, default=0, help='Alpha value for disentanglement');
parser.add_argument('--env_iteration', type=int, default=5,  help='Iterations of environment phase');

args = parser.parse_args();

## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## Load models
s = KeywordNet(**vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [100];

## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    s.__scheduler__.step()
        
## Evaluation code
if args.eval == True:
        
    sc, lab, trials = s.evaluateFromList(args.test_list, print_interval=100, test_path=args.test_path)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    
    dcf_c_miss = 1.
    dcf_c_fa = 1.
    dcf_p_target = 0.05

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, dcf_p_target, dcf_c_miss, dcf_c_fa)

    print('EER %2.4f, minDCF %.4f'%(result[1], mindcf))

    ## Save scores
    print('Type desired file name to save scores. Otherwise, leave blank.')
    userinp = input()

    while True:
        if userinp == '':
            quit();
        elif os.path.exists(userinp) or '.' not in userinp:
            print('Invalid file name %s. Try again.'%(userinp))
            userinp = input()
        else:
            with open(userinp,'w') as outfile:
                for vi, val in enumerate(sc):
                    outfile.write('%.4f %s\n'%(val,trials[vi]))
            quit();

## save code
pyfiles = glob.glob('./*.py')
strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

zipf = zipfile.ZipFile(result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
for file in pyfiles:
    zipf.write(file)
zipf.close()

f = open(result_save_path + '/run%s.cmd'%strtime, 'w')
f.write(' '.join(sys.argv))
f.close()

## Write args to scorefile
scorefile = open(result_save_path+"/scores.txt", "a+");

## Initialise data loader
trainLoader = get_data_loader(args.train_list, **vars(args));

while(1):   

    clr = [x['lr'] for x in s.__optimizer__.param_groups]

    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)));

    ## Train network
    loss, traineer = s.train_network(loader=trainLoader, alpha=args.alpha, num_steps=args.env_iteration);

    ## Validate and save
    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab, _ = s.evaluateFromList(args.test_list, print_interval=100, test_path=args.test_path)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print(args.save_path)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f"%( max(clr), traineer, loss, result[1]));
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f\n"%(it, max(clr), traineer, loss, result[1]));

        scorefile.flush()

        s.saveParameters(model_save_path+"/model%04d.model"%it);
        
        with open(model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
            eerfile.write('%.4f'%result[1])

        min_eer.append(result[1])

    else:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f"%( max(clr), traineer, loss));
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss));

        scorefile.flush()

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();