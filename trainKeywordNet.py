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

import sys, time, os, argparse
import numpy, random
import torch
import torch.nn as nn
import glob
import zipfile
import datetime
from tuneThreshold import *
from KeywordNet import KeywordNet
from DatasetLoader import get_data_loader
from DatasetLoader_classify import get_data_loader_classify

parser = argparse.ArgumentParser(description = "KeywordNet");

parser.add_argument('--config', type=str, default=None,  help='Config YAML file');
 
## Data loader
parser.add_argument('--batch_size', type=int, default=1,  help='Batch size');
parser.add_argument('--metric_batch_size', type=int, default=16,  help='Dictionary size');
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
parser.add_argument('--lr_step_size', type=float, default=1, help='Learning rate decaying step')

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="./data/test", help='Path for model and logs');

# Training and test data
parser.add_argument('--train_list',     type=str,   default="./train_test_lists/pretrain/train_list_margin_1s_filter_1000.txt",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="./train_test_lists/pretrain/test_list_margin_1s_filter_1000.txt",     help='Evaluation list');
parser.add_argument('--train_path',     type=str,   default="/mnt/scratch/datasets/keyword/words_filter_1s_cer20", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="/mnt/scratch/datasets/keyword/words_filter_1s_cer20", help='Absolute path to the test set');

# Noise data
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--musan_path',     type=str,   default="/mnt/datasets/speech_augmentation/musan_split", help='Absolute path to the test set');
parser.add_argument('--noise_path',     type=str,   default="/mnt/datasets/kws/speech_commands_v0.02/_background_noise_", help='Absolute path for the silence noise')

#Google speech dataset
parser.add_argument('--fine_train_list',     type=str,   default="./train_test_lists/finetune/fine_train_list.txt",     help='Train list');
parser.add_argument('--fine_train_path',     type=str,   default="/mnt/datasets/kws/speech_commands_v0.02/", help='Absolute path to the train set');
parser.add_argument('--fine_test_list',      type=str,   default="./train_test_lists/finetune/fine_test_list.txt",     help='Evaluation list');
parser.add_argument('--fine_test_path',      type=str,   default="/mnt/datasets/kws/speech_commands_v0.02/", help='Absolute path to the test set');
parser.add_argument('--test_acc_list',  type=str,   default="./train_test_lists/test/10_test_acc_list.txt", help='Evaluation Accuracy list')
parser.add_argument('--test_acc_path',  type=str,   default="/mnt/datasets/kws/speech_commands_v0.02/", help='Absolute path to the test accuracy set')

## Set target keywords
parser.add_argument('--target_keys','--list', nargs='+', default=None, help='Set keywords, e.g., --target_keys zero one two three ...')

## For enrollment
parser.add_argument('--enroll_list',    type=str,   default="./train_test_lists/enroll/10_enroll_list.txt", help='enroll list')
parser.add_argument('--enroll_path',    type=str,   default="/mnt/datasets/kws/speech_commands_v0.02/", help='Absolute path to the enroll set')
parser.add_argument('--enroll_num',     type=int,   default=10, help="number of shots")

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--n_maps',         type=int,   default=45,     help='Number of featuer maps');
parser.add_argument('--model',          type=str,   default="ResNet15",     help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=1001,    help='Embedding size in the last FC layer (the number of classes at training');
parser.add_argument('--nClasses',       type=int,   default=1001,    help='Number of classes to be classified')

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Only evaluation')

## For fine-tuning, add layer, freezing
parser.add_argument('--fine_tuning',        dest='fine_tuning',  action='store_true',  help='For fine tuning')
parser.add_argument('--sample_per_class',      type=int,   default=300,    help='Number of samples per class')

## Edited
parser.add_argument('--input_length',      type=int,  default=16000,  help='input length(default=16000)')
parser.add_argument('--seed',      type=int,  default=42,  help='seed number')

## Specific to environment removal
parser.add_argument('--alpha', type=float, default=0, help='Alpha value for disentanglement');
parser.add_argument('--env_iteration', type=int, default=5,  help='Iterations of environment phase');

## Remove silence or not
parser.add_argument('--no_silence', type=bool, default=False, help='If True, no silence during training')

args = parser.parse_args();

############################################
''' Setting '''
############################################
## Random seed 
def seed_everything(seed: int = 42):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True 

seed_everything(args.seed)

## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## Load models
s = KeywordNet(**vars(args));

## Set default target keywords
if args.target_keys == None:
    args.target_keys = '__silence__, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')
else:
    args.target_keys.append('__silence__')

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

if args.eval == True:
    pred, lab, sc, eer_lab = s.evaluateAccuracyFromList(args.target_keys, args.enroll_num, args.enroll_list, args.test_acc_list, enroll_path=args.enroll_path, test_path=args.test_acc_path, noise_path=args.noise_path)
    result = tuneThresholdfromScore(sc, eer_lab, [1, 0.1])
    f1, acc = f1_and_acc(pred, lab, None)
    print('EER %2.4f, FRR at FAR=2.5 %2.4f, FRR at FAR=10 %2.4f, F1-score %2.4f, Acc %2.4f'%(result[1], result[2], result[3], f1.mean(), acc))
    quit();

############################################
''' Train & Validation code '''
############################################
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
if args.fine_tuning == True:
    args.train_list = args.fine_train_list
    args.train_path = args.fine_train_path
    args.test_list = args.fine_test_list
    args.test_path = args.fine_test_path

print("Train list : %s"%args.train_list)

if args.trainfunc == 'softmax' or args.trainfunc == 'amsoftmax':
    trainLoader = get_data_loader_classify(args.train_list, **vars(args));
else:
    trainLoader = get_data_loader(args.train_list, **vars(args));

while(1):   

    clr = [x['lr'] for x in s.__optimizer__.param_groups]

    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)));

    ## Train network
    if args.trainfunc == 'softmax' or args.trainfunc == 'amsoftmax':
        loss, traineer = s.train_network_classify(loader=trainLoader)
    else:
        trainLoader.dataset.shuffle_dict()
        loss, traineer = s.train_network(epoch=it, loader=trainLoader, alpha=args.alpha, num_steps=args.env_iteration);

    ## Validate and save
    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");
        pred, lab, sc, eer_lab = s.evaluateAccuracyFromList(args.target_keys, args.enroll_num, args.enroll_list, args.test_acc_list, enroll_path=args.enroll_path, test_path=args.test_acc_path, noise_path=args.noise_path)
        result = tuneThresholdfromScore(sc, eer_lab, [1, 0.1]);
        f1, acc = f1_and_acc(pred, lab, None)

        print(args.save_path)
        print('EER %2.4f, FRR at FAR=2.5 %2.4f, FRR at FAR=10 %2.4f, F1-score %2.4f, Acc %2.4f'%(result[1], result[2], result[3], f1.mean(), acc))
        
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, Accuracy %2.4f, F1-score %2.4f, FRR@FAR=2.5 %2.4f, FRR@FRR=10 %2.4f\n"%(it, max(clr), traineer, loss, result[1], acc, f1.mean(), result[2], result[3]));
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
