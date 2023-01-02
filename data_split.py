import sys, time, argparse
import re
from pathlib import Path

import os
import glob

import random

random.seed(0)

parser = argparse.ArgumentParser(description = "Data Divider")

parser.add_argument('--dataset_path', type=str, default='/mnt/work4/datasets/keyword/', help='parent path of dataset directory')
parser.add_argument('--save_path', type=str, default='set_list', help='save path')
parser.add_argument('--dataset_name', type=str, default='speech_commands_v0.02', help='dataset (directory) name')
args = parser.parse_args()

root = args.dataset_path + args.dataset_name
save_root = args.save_path

## for Google Speech Commands
DATA_CONFIG = {'pre_defined': 'yes, no, up, down, left, right, on, off, stop, go'.split(', '),
			   'user_defined': 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', '),
			   'unknown' : 'bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow, forward, backward, follow, learn, visual'.split(', ')}

num_test	= 10 # number of test keywords 

## rand_sample SHOULD BE UNDER 190 (i.e. ~189) because of the number of positive samples ##
rand_sample = 50 # number of pos & neg samples per one keyword 
rand_select = 10 # number of randomly selected anchor wav

num_audios  = 1500
num_enroll  = 5

num_test_acc = 300 ## 10/6 300->3000
num_test_unknown = num_test_acc // len(DATA_CONFIG['unknown'])

def key_list(typ, data_config=DATA_CONFIG, root=root):
	tot_list = {}

	assert os.path.isdir(root)

	for d in os.listdir(root):
		if d in data_config[typ]:
			tot_list[d] = []
			for wav_f in os.listdir(os.path.join(root, d)):
				tot_list[d].append(wav_f)

	return tot_list

def make_enroll_list(user_keys, unknown_keys):
	f_enroll = open(os.path.join(save_root, '5_300_enroll_list.txt'), 'w')

	enroll_dict = {}

	user_evicted = {}
	unknown_evicted  = {}

	for key, audios in user_keys.items():
		enroll_audios = random.sample(audios, num_enroll)
		enroll_dict[key] = enroll_audios

		evicted_list = [w for w in audios if w not in enroll_audios]
		user_evicted[key] = evicted_list

	for key, audios in unknown_keys.items():
		enroll_audios = random.sample(audios, num_enroll)
		enroll_dict[key] = enroll_audios

		evicted_list = [w for w in audios if w not in enroll_audios]
		unknown_evicted[key] = evicted_list

	for key, audios in enroll_dict.items():
		for audio in audios:
			f_enroll.write(key + ' ' + key + '/' + audio + '\n')

	f_enroll.close()

	return (user_evicted, unknown_evicted)

def make_test_list(user_keys, unknown_keys):
	f_test = open(os.path.join(save_root, '5_300_test_acc_list.txt'), 'w')

	test_dict = {}

	unknown_evicted = {}

	for key, audios in user_keys.items():
		test_audios = random.sample(audios, num_test_acc)
		test_dict[key] = test_audios

	for key, audios in unknown_keys.items():
		# print(audios)
		test_audios = random.sample(audios, num_test_unknown)
		test_dict[key] = test_audios

		evicted_list = [w for w in audios if w not in test_audios]
		unknown_evicted[key] = evicted_list

	test_list = []
	for key, audios in test_dict.items():
		for audio in audios:
			test_list.append(key + '/' + audio)

	random.shuffle(test_list)

	for audio in test_list:
		f_test.write(audio.split('/')[0] + ' ' + audio + '\n')

	f_test.close()

	return unknown_evicted

def make_fine_tune_list(pre_keys, unknown_keys):
	f_train = open(os.path.join(save_root, 'fine_tune_list.txt'), 'w')

	fine_dict = {}

	for key, audios in pre_keys.items():
		train_audios = random.sample(audios, num_audios)
		fine_dict[key] = train_audios

	for key, audios in unknown_keys.items():
		train_audios = random.sample(audios, num_audios)
		fine_dict[key] = train_audios

	for key, audios in fine_dict.items():
		for audio in audios:
			f_train.write(key + ' ' + key + '/' + audio + '\n')

	f_train.close()

def make_eer_test_list(data_config=DATA_CONFIG):
	f_test   = open(os.path.join(save_root, 'test_list.txt'), 'w')
	keys     = data_config['user_defined']
	f_dict 	 = {}
	anc_dict = {}
	for d in os.listdir(root):
		if d in keys:
			_wav_f_list = os.listdir(os.path.join(root, d))
			wav_f_list  = [d+'/'+item for item in _wav_f_list]
			rand_f_list = random.sample(wav_f_list, rand_select)

			f_dict[d]   = [item for item in wav_f_list if item not in rand_f_list]
			anc_dict[d] = rand_f_list

	for key in keys:
		pos_samples = []
		neg_samples = []

		pos_samples = random.sample(f_dict[key], rand_sample * (num_test - 1))
		neg_samples = [random.sample(v, rand_sample) for k, v in f_dict.items() if k != key]
		neg_samples = [item for sublist in neg_samples for item in sublist]
		neg_samples = random.sample(neg_samples, len(neg_samples))

		pos_pairs = []
		neg_pairs = []
		for anc_f in anc_dict[key]:
			pos_pairs.append([('1', anc_f, pos_sample) for pos_sample in pos_samples])
			neg_pairs.append([('0', anc_f, neg_sample) for neg_sample in neg_samples])

		pos_pairs = [item for sublist in pos_pairs for item in sublist]
		neg_pairs = [item for sublist in neg_pairs for item in sublist]

		whole_pairs = [x for y in zip(pos_pairs, neg_pairs) for x in y]

		for pair in whole_pairs:
			f_test.write(pair[0]+' '+pair[1]+' '+pair[2]+'\n')

	f_test.close()

def main():
	user_keys = key_list('user_defined')
	pre_keys  = key_list('pre_defined')
	unknown_keys = key_list('unknown')

	user_keys, unknown_keys = make_enroll_list(user_keys, unknown_keys)
	unknown_keys = make_test_list(user_keys, unknown_keys)

	make_fine_tune_list(pre_keys, unknown_keys)

main()
# make_eer_test_list()