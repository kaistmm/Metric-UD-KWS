import sklearn.metrics as metrics
import numpy as np
import os, argparse
import csv

parser = argparse.ArgumentParser(description = "Plot ROC curve")

parser.add_argument('--np_path', type=str, default='np_results/ICASSP')
parser.add_argument('--exp_names', nargs='+', default=['GSC_SOFT', 'SOFT_AP', 'AMSOFT_AP', 'AP_AP'], help='exp names')
parser.add_argument('--plot_file', type=str, default='ROC_curve.png')

args = parser.parse_args()

csv_path = 'base_paper_roc.csv'

np_path = args.np_path
exp_names = args.exp_names

from matplotlib import ticker
from matplotlib import pyplot as plt

lines = ['-', '-', '-', '-']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
legends = ['Softmax w/o PT (EER: 7.79%)', 'PT: Softmax, FT: AP (EER: 3.77%)',
		   'PT: AM-Soft, FT: AP (EER: 4.80%)', 'PT: AP, FT: AP (EER: 3.24%)']

def DETCurve(far_frrs, lines=lines, legends=legends):
	fig,ax = plt.subplots()
	plt.title('Detection Error Tradeoff', fontsize=13)
	i=0
	for far_frr in far_frrs:
		plt.plot(far_frr[0]*100.0, far_frr[1]* 100.0, colors[i], linestyle=lines[i], label=legends[i])
		i += 1
	plt.legend(loc = 'upper right', prop={'size':8})

	plt.xscale('log')
	plt.yscale('log')
	x_ticks_to_use = [1, 2, 5, 10, 20, 50]
	y_ticks_to_use = [1, 2, 5, 10, 20, 50]
	# ticks_to_use = [1, 5, 20, 50, 80, 95, 99]
	ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
	ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
	ax.set_xticks(x_ticks_to_use)
	ax.set_yticks(y_ticks_to_use)

	lims = [
    	np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    	np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]

	# now plot both limits against eachother
	ax.plot(lims, lims, 'k', linestyle='dashed', alpha=0.75, zorder=0, linewidth=0.7)
	# ax.set_xticks(ticks_to_use)
	# ax.set_yticks(ticks_to_use)
	plt.axis([1, 50, 1, 50])
	plt.minorticks_off()
	plt.xlabel('False Alarm Rate (%)', fontsize=12)
	plt.ylabel('False Rejection Rate (%)', fontsize=12)
	plt.savefig('DET_curve.png')


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


DETCurve(far_frrs)

# # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')