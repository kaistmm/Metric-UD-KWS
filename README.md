# Metric learning for user defined KWS
Official code for Metric learning for user-defined keyword spotting

[METRIC LEARNING FOR USER-DEFINED KEYWORD SPOTTING](https://arxiv.org/pdf/2211.00439.pdf)

[Project page](https://mm.kaist.ac.kr/projects/kws/)

### Introdcution



### License


### Installation
<code>pin install -r requirements.txt</code>.

### Dataset (LibriSpeech Keywords)
Please find the LibriSpeech Keyworkds(LSK) [here]().

### Pre-train

<code>$python trainKeywordNet.py --save_path [save_path] --augment True --dict_size [dict_size] --trainfunc [trainfunc] --model [ResNet15, ResNet26]</code>.

### Fine-tune

<code>$python trainKeywordNet.py --save_path [save_path] --augment True --dict_size 26 --batch_size 1 --trainfunc [trainfunc] --fine_tunning True --initial_model [model.pt] --lr 0.00001 --lr_step_size 1</code>.
