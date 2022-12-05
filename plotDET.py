import sklearn.metrics as metrics
import numpy as np
import os, argparse
import csv

parser = argparse.ArgumentParser(description = "Plot ROC curve")

parser.add_argument('--np_path', type=str, default='np_results/ICASSP')
parser.add_argument('--exp_names', nargs='+', default=['500_500', '500_1000', '1000_500', 'AP_AP', 'ENKR'], help='exp names')
parser.add_argument('--plot_file', type=str, default='ROC_curve.png')

args = parser.parse_args()

csv_path = 'base_paper_roc.csv'

np_path = args.np_path
exp_names = args.exp_names

from matplotlib import ticker
from matplotlib import pyplot as plt

lines = ['-.', '-.', '-.', '-.', '-.']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
legends = ['# Classes: 500, # Samples: 500 (EER: 4.13%)', '# Classes: 500, # Samples: 1,000 (EER: 3.94%)',
		   '# Classes: 1,000, # Samples: 500 (EER: 3.63%)', '# Classes: 1,000, # Samples: 1,000 (EER: 3.24%)',
		   '# Classes: 2,000 (LSK+KSK), # Samples: 1,000 (EER: 3.07%)']

def DETCurve(far_frrs, lines=lines, legends=legends):
	fig,ax = plt.subplots()
	plt.title('Detection Error Tradeoff', fontsize=13)
	i=0

	for far_frr in far_frrs:
		plt.plot(far_frr[0]*100.0, far_frr[1]*100.0, colors[i], linestyle=lines[i], label=legends[i])
		i += 1
	plt.legend(loc = 'upper right', prop={'size':8})

	plt.xscale('log')
	plt.yscale('log')
	x_ticks_to_use = [1, 2, 5, 10, 20]
	y_ticks_to_use = [1, 2, 5, 10, 20]
	# ticks_to_use = [1, 5, 20, 50, 80, 95, 99]
	ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
	ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
	ax.set_xticks(x_ticks_to_use)
	ax.set_yticks(y_ticks_to_use)

	lims = [
    	np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    	np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]
	# import pdb; pdb.set_trace()
	ax.plot(lims, lims, 'k', linestyle='dashed', alpha=0.75, zorder=0, linewidth=0.7)

	# i = 0
	# for far_frr in far_frrs:
	# 	import pdb; pdb.set_trace()
	# 	f = np.linspace(lims[0], lims[1], far_frr[0].shape[0])
	# 	idx = np.argwhere(np.diff(np.sign(f - far_frr[0]*100.0))).flatten()
	# 	plt.plot(far_frr[0][idx]*100.0, f[idx], 'ro')
	# 	i += 1

	# now plot both limits against eachother
	
	# ax.set_xticks(ticks_to_use)
	# ax.set_yticks(ticks_to_use)
	plt.axis([1, 20, 1, 20])
	plt.minorticks_off()
	plt.xlabel('False Alarm Rate (%)', fontsize=12)
	plt.ylabel('False Rejection Rate (%)', fontsize=12)
	plt.savefig('DET_curve_2.png')


 #    i = 0
	# for far_frr in far_frrs:
	# 	# import pdb; pdb.set_trace()
	# 	plt.plot(far_frr[0], far_frr[1], 'k', linestyle=lines[i], label=legends[i])
	# 	i += 1
 #    fig,ax = plt.subplots()
 #    plot(fps,fns)
 #    yscale('log')
 #    xscale('log')
 #    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
 #    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
 #    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
 #    ax.set_xticks(ticks_to_use)
 #    ax.set_yticks(ticks_to_use)
 #    axis([0.001,50,0.001,50])

def csv2list(csv_path=csv_path):
	default_far = []
	default_frr = []
	with open(csv_path, 'r') as vox1_csv:
		rdr = csv.reader(vox1_csv)
		for line in rdr:
			default_far.append(float(line[0]))
			default_frr.append(float(line[1]))

	return (default_far, default_frr)


far_frr_paths = []
for exp in exp_names:
	far_path = np_path + '/' + exp + '_far.npy'
	frr_path = np_path + '/' + exp + '_frr.npy'
	far_frr_paths.append((far_path, frr_path))

far_frrs = []
for path in far_frr_paths:
	far_frrs.append((np.load(path[0]), np.load(path[1])))

# default_far, default_frr = csv2list(csv_path)
# del default_far[-2]
# del default_far[-2]

# del default_frr[-2]
# del default_frr[-2]

# import pdb; pdb.set_trace()

DETCurve(far_frrs)

# method I: plt
# import matplotlib.pyplot as plt
# plt.figure(figsize=(100, 60))
# plt.title('Detection Error Tradeoff (DET) curves', fontsize=13)
# # plt.plot(np.array(default_far), np.array(default_frr), 'b', linestyle='-', label='[17] with incremental training')
# i = 0
# for far_frr in far_frrs:
# 	# import pdb; pdb.set_trace()
# 	plt.plot(far_frr[0], far_frr[1], 'k', linestyle=lines[i], label=legends[i])
# 	i += 1
# plt.legend(loc = 'upper right')
# # plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 0.1])
# plt.ylim([0, 0.2])
# plt.xlabel('False Alarm Rate', fontsize=12)
# plt.ylabel('False Rejection Rate', fontsize=12)
# plt.savefig('DET_curve_2.png')

# method II: ggplot
# from ggplot import *
# df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
# ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

# import sklearn.metrics as metrics
# import numpy as np
# import os, argparse

# parser = argparse.ArgumentParser(description = "Plot ROC curve")

# parser.add_argument('--np_path', type=str, default='np_results/ICASSP')
# parser.add_argument('--exp_names', nargs='+', default=['500_500', '500_1000', '1000_500', 'AP_AP', 'ENKR'], help='exp names')
# parser.add_argument('--plot_file', type=str, default='ROC_curve.png')

# args = parser.parse_args()

# np_path = args.np_path
# exp_names = args.exp_names

# far_frr_paths = []
# for exp in exp_names:
# 	far_path = np_path + '/' + exp + '_far.npy'
# 	frr_path = np_path + '/' + exp + '_frr.npy'
# 	far_frr_paths.append((far_path, frr_path))

# far_frrs = []
# for path in far_frr_paths:
# 	far_frrs.append((np.load(path[0]), np.load(path[1])))

# # import pdb; pdb.set_trace()

# lines = ['-', '--', '-.', ':', dotted]

# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# i = 0
# for far_frr in far_frrs:
# 	plt.plot(far_frr[0], far_frr[1], 'b', linestyle=lines[i], label=exp_names[i])
# 	i += 1
# plt.legend(loc = 'upper right')
# # plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 0.2])
# plt.ylim([0, 0.5])
# plt.xlabel('False Alarm Rate (FAR)')
# plt.ylabel('False Rejection Rate (FRR)')
# plt.savefig('testROC.png')

# # method II: ggplot
# # from ggplot import *
# # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
# # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')