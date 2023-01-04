## User-defined Wake-up-word Detection

This repository contains the baseline code for wake-up-word detection by using metric learning. Our baseline code is based on the code [voxceleb trainer](https://github.com/clovaai/voxceleb_trainer), which is implemented for the speaker recognition task.

#### Dependencies
```
conda create --file requirements.txt -n [env_name] -c pytorch -c conda-forge
```

#### Data preparation

The [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) datasets are used for these experiments. Follow the instructions on [this page](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html) to download and prepare the data for training. We used Speech Commands **v0.01** (30 keywords in total) for our baseline.

The dataset is split into train dataset and test dataset. Train dataset is comprised with 20 kinds of keywords out of 30. Test dataset is comprised with the other 10 kinds of keywords. Lists for both of the split dataset are in the directory of 'dataset_split'. 

#### Data preparation for LSK
- Step 1. Run ```LSK_preprocess.py```.
- Step 2. Run ```LSK_make_list.py```.

#### Pre-trained models
- Pre-trained on LSK & Fine-tuned on GSC (both using angleproto loss)
```
PT_models/PTAP_FTAP_en.model
```

- Pre-trained on LSK+KSK & Fine-tuned on GSC (both using angleproto loss)
```    
PT_models/PTAP_FTAP_en_kr.model
```

#### Training examples
- Pre-training (Softmax, AM-Softmax)
```
python trainKeywordNet.py --save_path [save_path] --augment True --batch_size [batch_size] --trainfunc [trainfunc] --model [ResNet15, ResNet26]
```
- Pre-training (Prototypical, Angular Prototypical)
```
python trainKeywordNet.py --save_path [save_path] --augment True --metric_batch_size [metric_batch_size] --batch_size 1 --trainfunc [trainfunc] --model [ResNet15, ResNet26]
```

- Fine-tuning (Softmax, AM-Softmax)
```
python trainKeywordNet.py --fine_tuning --save_path [save_path] --augment True --batch_size [batch_size] --trainfunc [trainfunc] --model [ResNet15, ResNet26] --initial_model [model.pt] --lr 0.00001
```
- Fine-tuning (Prototypical, Angular Prototypical)
```
python trainKeywordNet.py --fine_tuning --save_path [save_path] --augment True --metric_batch_size [metric_batch_size] --batch_size 1 --trainfunc [trainfunc] --model [ResNet15, ResNet26] --initial_model [model.pt] --lr 0.00001
```
#### Testing Examples
```
python trainKeywordNet.py --eval --save_path [save_path] --initial_model [initial_model] --enroll_num [1, 5, 10 (# of shots)] --enroll_list [/path/to/enroll_list.txt] --test_acc_list [/path/to/test_acc_list.txt]
```

#### Implemented loss functions
```
Softmax (softmax)
Additive Margin Softmax (amsoftmax)
Prototypical (proto)
Angular Prototypical (angleproto)
```

#### Implemented models
For the model, res15 from *"deep residual learning for small-footprint keyword spotting", R. Tang & J. Lin, 2018, ICASSP* is used. Code for the model is based on [Honk: CNNs for Keyword Spotting](https://github.com/castorini/honk).
```
ResNet15
ResNet26
```
