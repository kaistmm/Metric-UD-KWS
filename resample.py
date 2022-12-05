import os, shutil
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

target_dir = '/mnt/scratch/datasets/new_enkr'
new_path = '/mnt/scratch/datasets/enkr_resample'

if not os.path.exists(new_path):
	os.makedirs(new_path)

for d in os.listdir(target_dir):
# for d in tqdm(os.listdir(target_dir)):
	d_path = os.path.join(target_dir, d)

	if not os.path.isdir(d_path):
		continue
	new_dir = os.path.join(new_path, d)
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

	for f in os.listdir(d_path):
		f_path = os.path.join(d_path, f)
		_, f_ext = os.path.splitext(f_path)

		if f_ext != '.wav':
			continue

		save_path = os.path.join(new_dir, f)

		# if not d.isalpha():
			# print(d)
		waveform, sample_rate = torchaudio.load(f_path, normalize=True)

		if sample_rate == 16000:
			print(d)
			# shutil.copyfile(f_path, save_path)
			continue

		transform = T.Resample(sample_rate, 16000)
		waveform = transform(waveform)
		torchaudio.save(save_path, waveform, sample_rate)