import os, pdb
import operator
import pickle
from tqdm import tqdm

CHECK_WORDS = 1000
# path = '/mnt/scratch/datasets/words_filtered/'
path = '/mnt/work4/datasets/keyword/words_filter_1s_cer20/'
# path = '/mnt/scratch/datasets/audio/train/de/clips/'
# path = '/mnt/scratch2/jjm/Cmd_General/KSC/' # 3000 words
# path = '/mnt/scratch/datasets/keyword/en_kr/'

folder_list = os.listdir(path) 
num_folder = len(folder_list)
# print(num_folder)

num_words = {}
delete_list={}
num_three = 0

for folder in tqdm(folder_list):
    folder_path = path + folder 

    if os.path.isdir(folder_path):
        file_list = os.listdir(folder_path)
    num_file = len(file_list)

    ''' Remove words which have only 1 characters '''
    if len(folder) > 1:
        num_words[folder] = num_file
    # num_words[folder] = num_file

    ''' Remove empty folders '''
    # if num_file == 0:
        # delete_list[folder] = num_file
        # os.rmdir(path+folder)

# operator.itemgetter 
num_words_sorted = sorted(num_words.items(), key=operator.itemgetter(1), reverse=True) 

## Remove 20 words
num_words_sorted = num_words_sorted[13:]

USER_WORDS = ['ZERO','ONE', 'ONES', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
num_total_words = 0

for i in range(CHECK_WORDS+10):
    if num_words_sorted[i][0] not in USER_WORDS: 
        num_total_words += num_words_sorted[i][1]

    else : 
        del num_words_sorted[i]
    # if num_words_sorted[i][1] >= 1000:
        # num_total_words += 1000
    # else:
        # num_total_words += num_words_sorted[i][1]

print(num_words_sorted[:CHECK_WORDS])
print(num_total_words)

os.makedirs('./words_list', exist_ok=True)
with open('./words_list/words_filter_0.1s_cer50_list.pkl', 'wb') as f:
    pickle.dump(num_words_sorted,f)

