import sklearn.metrics as metrics
import numpy as np
import os, argparse

parser = argparse.ArgumentParser(description = "Plot ROC curve")

parser.add_argument('--np_path', type=str, default='np_results')
parser.add_argument('--exp_names', nargs='+', default=['softmax', 'amsoftmax', 'ap'], help='exp names')
parser.add_argument('--plot_file', type=str, default='ROC_curve.png')

args = parser.parse_args()

np_path = args.np_path
exp_names = args.exp_names

far_frr_paths = []
for exp in exp_names:
	far_path = np_path + '/' + exp + '_far.npy'
	frr_path = np_path + '/' + exp + '_frr.npy'
	far_frr_paths.append((far_path, frr_path))

far_frrs = []
for path in far_frr_paths:
	far_frrs.append((np.load(path[0]), np.load(path[1])))

# import pdb; pdb.set_trace()

lines = ['-', '--', '-.', ':', '.']

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
i = 0
for far_frr in far_frrs:
	plt.plot(far_frr[0], far_frr[1], 'b', linestyle=lines[i], label=exp_names[i])
	i += 1
plt.legend(loc = 'upper right')
# plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 0.2])
plt.ylim([0, 0.5])
plt.xlabel('False Alarm Rate (FAR)')
plt.ylabel('False Rejection Rate (FRR)')
plt.savefig('testROC.png')

# method II: ggplot
# from ggplot import *
# df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
# ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')