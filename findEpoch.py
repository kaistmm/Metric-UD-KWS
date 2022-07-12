import sys, os, argparse
# import yaml
import numpy as np
import pdb
import zipfile
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description = "Plot the results")

parser.add_argument('--data_path', type=str, default='data', help='where the data at')
parser.add_argument('--exp_names', nargs='+', default=['exp1'], help='exp names')
parser.add_argument('--max_epoch', type=int, default=150, help='max epoch')
parser.add_argument('--test_interval', type=int, default=3, help='text interval')
parser.add_argument('--teer_file', type=str, default='TEER.png', help='TEER file name')
parser.add_argument('--tloss_file', type=str, default='TLOSS.png', help='TLOSS file name')
parser.add_argument('--veer_file', type=str, default='VEER.png', help='VEER file name')

args = parser.parse_args()

data_path = Path(args.data_path)
exp_names = args.exp_names
max_epoch = args.max_epoch
test_interval = args.test_interval

teer_file = args.teer_file
tloss_file = args.tloss_file
veer_file = args.veer_file

paths = []
for exp in exp_names:		
	path = data_path / exp / 'result' / 'scores.txt'
	if not path.exists():
		sys.stderr.write("No such path.\n")
	paths.append(path)

VEERs = []

for path in paths:
	with open(path, 'r') as f:
		VEER = []
		for line in f:
			tokens = [i for i in line.split()]
			if len(tokens) >= 10:
				VEER.append(float(tokens[9]))
		if len(VEER) != 0: VEERs.append(VEER)

for pair in zip(exp_names, VEERs):
	print("exp name : {}, min VEER epoch : {}".format(pair[0], 1 + np.argsort(np.array(pair[1]))[:10]))
