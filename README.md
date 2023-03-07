# Metric learning for user-defined KWS
This repository contains the official code for Metric learning for user-defined Keyword spotting. Our code is based on the code voxceleb trainer, which is implemented for the speaker recognition task.

[METRIC LEARNING FOR USER-DEFINED KEYWORD SPOTTING](https://arxiv.org/pdf/2211.00439.pdf)

[Project page](https://mm.kaist.ac.kr/projects/kws/)


### Data preparation
#### LibriSpeech Keywords
Please find the LibriSpeech Keyworkds(LSK) [here]().
#### Google Speech Commands
The Google Speech Commands datasets are used for these experiments. Follow the instructions on this page to download and prepare the data for training. We used Speech Commands v0.01 (30 keywords in total) for our baseline.


### Train a new model
#### Dependencies
```sh
$ conda create --file requirements.txt -n [env_name] -c pytorch -c conda-forge
```

#### Pre-train
```sh
$ python trainKeywordNet.py --save_path [save_path] --augment True --dict_size [dict_size] --trainfunc [trainfunc] --model [ResNet15, ResNet26]
```

#### Fine-tune
```sh
$ python trainKeywordNet.py --save_path [save_path] --augment True --dict_size 16 --trainfunc [trainfunc] --fine_tunning True --initial_model [model.pt] --lr 0.00001 --lr_step_size 1
```

#### Implemented loss functions
```
Softmax (Softmax)
Additive Margin Softmax loss(AM-Soft)
Angular Prototypical (angleproto)
```

#### Implemented models
For the model, res15 from *"deep residual learning for small-footprint keyword spotting", R. Tang & J. Lin, 2018, ICASSP* is used. Code for the model is based on [Honk: CNNs for Keyword Spotting](https://github.com/castorini/honk).
```
ResNet15
ResNet26
```
