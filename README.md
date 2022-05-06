## User-defined Wake-up-word Detection

This repository contains the baseline code for wake-up-word detection by using metric learning. Our baseline code is based on the code [voxceleb trainer](https://github.com/clovaai/voxceleb_trainer), which is implemented for the speaker recognition task.

#### Dependencies
```
conda create --file requirements.txt -n [env_name] -c pytorch -c conda-forge
```

#### Data preparation

The [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) datasets are used for these experiments. Follow the instructions on [this page](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html) to download and prepare the data for training. We used Speech Commands **v0.01** (30 keywords in total) for our baseline.

The dataset is split into train dataset and test dataset. Train dataset is comprised with 20 kinds of keywords out of 30. Test dataset is comprised with the other 10 kinds of keywords. Lists for both of the split dataset are in the directory of 'dataset_split'. 

#### Training examples

- Default model
```
python ./trainKeywordNet.py --save_path data/[exp_name] --train_path [/path/to/speech_commands_v0.01] --test_path [/path/to/speech_commands_v0.01]
```

#### Implemented loss functions
```
Prototypical (proto)
Angular Prototypical (angleproto)
```

#### Implemented models
For the model, res15 from *"deep residual learning for small-footprint keyword spotting", R. Tang & J. Lin, 2018, ICASSP* is used. Code for the model is based on [Honk: CNNs for Keyword Spotting](https://github.com/castorini/honk).
```
ResNet15
```