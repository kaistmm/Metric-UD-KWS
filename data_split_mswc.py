import argparse, pickle, os, random

random.seed(0)

parser = argparse.ArgumentParser(description = "Data Divider")

parser.add_argument('--dataset_path', type=str, default='/mnt/work4/datasets/keyword/', help='parent path of dataset directory')
parser.add_argument('--save_path', type=str, default='./data_split', help='save path')
parser.add_argument('--dataset_name', type=str, default='MSWC', help='dataset (directory) name')
parser.add_argument('--language', type=str, default='english')
args = parser.parse_args()

lang2code = {'english': 'en', 'spanish': 'es', 'chinese': 'zh-CN', 'italian': 'it', 'polish': 'pl'}

root = args.dataset_path + args.dataset_name
save_root = args.save_path
audio_path = os.path.join(root, args.language, lang2code[args.language], 'clips_wav')
pkl_path = os.path.join(root, f"{lang2code[args.language]}_list.pkl")
with open(pkl_path, 'rb') as f:
    keyword2wav = pickle.load(f)

keywords = list(keyword2wav.keys())
# DATA_CONFIG = {'pre_defined': keywords[15:25], 'user_defined': keywords[25:35], 'unknown': keywords[:15]}
DATA_CONFIG = {'pre_defined': keywords[25:35], 'user_defined': keywords[15:25], 'unknown': keywords[:15]}

num_test	= 10 # number of test keywords 

num_audios  = 800
num_enroll  = 10

num_test_acc = 300 ## 10/6 300->3000
num_test_unknown = num_test_acc // len(DATA_CONFIG['unknown'])

enroll_f_name = f"{num_enroll}_{args.dataset_name}_{lang2code[args.language]}_enroll_list.txt"
test_f_name   = f"{num_enroll}_{args.dataset_name}_{lang2code[args.language]}_test_acc_list.txt"
ft_f_name     = f"{num_enroll}_{args.dataset_name}_{lang2code[args.language]}_fine_tune_list.txt"

def key_list(typ, data_config=DATA_CONFIG, root=audio_path):
	tot_list = {}

	assert os.path.isdir(root)

	for d in os.listdir(root):
		if d in data_config[typ]:
			tot_list[d] = []
			for wav_f in os.listdir(os.path.join(root, d)):
				if wav_f.split('.')[-1] == 'wav':
					tot_list[d].append(wav_f)

	return tot_list

def make_enroll_list(user_keys, unknown_keys):
	f_enroll = open(os.path.join(save_root, enroll_f_name), 'w')

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
	f_test = open(os.path.join(save_root, test_f_name), 'w')

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

	# random.shuffle(test_list)

	for audio in test_list:
		f_test.write(audio.split('/')[0] + ' ' + audio + '\n')

	f_test.close()

	return unknown_evicted

def make_fine_tune_list(pre_keys, unknown_keys):
	f_train = open(os.path.join(save_root, ft_f_name), 'w')

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

def main():
	print(', '.join(DATA_CONFIG['user_defined']))
	user_keys = key_list('user_defined')
	pre_keys  = key_list('pre_defined')
	unknown_keys = key_list('unknown')
	# import pdb; pdb.set_trace()
	user_keys, unknown_keys = make_enroll_list(user_keys, unknown_keys)
	unknown_keys = make_test_list(user_keys, unknown_keys)

	make_fine_tune_list(pre_keys, unknown_keys)

main()