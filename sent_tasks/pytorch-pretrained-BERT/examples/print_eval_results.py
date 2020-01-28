import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   
from matplotlib.pylab import plt

# python print_eval_results.py -p -t single_domain -m pretrained
# model: pretrained, biased_qnli, biased_sst-2

XLABELS = {"size": "size", 
	"allsize": "percentage of WikiText-2", 
	"domain": "no. of domains", 
	"single_domain": "domain", 
	"accdomain": "number of domains",
	"moredomain": "number of domains"}
TITLES = {"size": "different sizes", 
	"allsize": "size of templates",
	"accdomain": "number of domains",
	"domain": "accumulating domains", 
	"single_domain": "different domains",
	"moredomain": "number of domains"}
MODEL_FORMAL_NAMES = {
	"biased_qnli": "BERT fine-tuned on QNLI", 
	"biased_sst-2": "BERT fine-tuned on SST-2",
	"biased_cola": "BERT fine-tuned on CoLA",
	"pretrained": "pretrained BERT"}

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_type","-t", required=True, 
		choices=["size", "domain", "single_domain", "allsize", "accdomain", "moredomain"])
	parser.add_argument("--model","-m", required=True)
	args = parser.parse_args()
	args.eval_results_dir = "aaai_bias_eval_results"
	args.model_results_dir = os.path.join(args.eval_results_dir, args.model)
	return args

def print_results(collected_results, versions):
	# Print results
	print(" "*5, end="")
	for version in versions:
		print("{:>12}".format(version), end="")
	print()
	for test_name in collected_results:
		test_results = collected_results[test_name]
		print("{:<5}".format(test_name), end="")
		for version in versions:
			print("{:12.3f}".format(test_results[version]['mean']), end="")
		print()

def print_all_results(collected_results, versions, args):
	test_type = args.test_type
	model_name = args.model

	for test_name in collected_results:
		test_results = collected_results[test_name]
		x, y = [], []
		for version in versions:
			if (version in test_results):
				x.append(version)
				y.append(test_results[version]['mean'])
		plt.plot(x, y, label=test_name)
	
	plt.xticks(np.arange(len(x)), x)
	plt.xlabel(XLABELS[test_type])
	plt.ylabel('average absolute effect size')
	plt.legend(loc='best')
	plt.title("SEAT effect sizes on {} with {}".format(model_name, TITLES[test_type]))
	plot_path = os.path.join(args.eval_results_dir, "plots",
		"{}-{}.png".format(model_name, test_type))
	plt.savefig(plot_path)

# average over all tests
def plot_average(collected_results, versions, args, plot_std=True):
	test_type = args.test_type
	model_name = args.model

	means, stds = [], []
	for version in versions:
		data = collected_results[version]
		if (plot_std):
			means.append(np.mean(data))
			stds.append(np.std(data))
		else:
			means.append(data)

	means = np.array(means)
	stds = np.array(stds)
	if (test_type == "size" or test_type == "allsize"):
		x = ["0%", "20%", "40%", "60%", "80%", "100%"]
	elif (test_type == "accdomain" or test_type == "moredomain"):
		x = [0, 1, 2, 3, 4]
	else:
		x = versions

	color = 'blue'
	plt.plot(x, means, color=color)
	if (plot_std):
		plt.fill_between(x, means-stds, means+stds, 
			alpha=0.1, edgecolor=color, facecolor=color,
			linewidth=1, antialiased=True)

	plt.xticks(np.arange(len(x)), x, fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlabel(XLABELS[test_type], fontsize=18)
	plt.ylabel('average absolute effect size', fontsize=18)
	plt.title("Influence of {} on bias removal \nfor {}".format(TITLES[test_type], MODEL_FORMAL_NAMES[model_name]),
		fontsize=18)
	plt.tight_layout()

	plot_path = os.path.join(args.eval_results_dir, "plots",
		"{}-{}-avg{}.png".format(model_name, test_type, "-std" if plot_std else ""))
	plt.savefig(plot_path)

def get_versions(test_type):
	if (test_type in ['size', 'domain', "allsize", "accdomain", "moredomain"]):
		versions = ["biased"]
		for i in range(1, 6):
			version = version = "{}{}".format(test_type, i)
			versions.append(version)
		return versions
	elif (test_type == "single_domain"):
		versions = ['biased', "news", "reddit", "sst", "pom", "wikitext"]
		return versions
	else:
		raise Exception("Invalid test type")

def load_from_json(path, average):
	results = json.load(open(path, 'r'))
	if (average):
		for test_name in results:
			test_results = results[test_name]
			for metric in test_results:
				data_points = test_results[metric]
				mean = np.mean(data_points)
				std = np.std(data_points)
				test_results[metric] = {"mean": mean, "std": std}
	else:
		for test_name in results:
			test_results = results[test_name]
			for metric in test_results:
				data_point = test_results[metric]
				test_results[metric] = {"mean": data_point}
	return results

def load_and_avg_over_tests(path):
	results = json.load(open(path, 'r'))
	avg_results = dict()
	num_tests = len(results)
	for test_name in results:
		test_results = results[test_name]
		for metric in test_results:
			data = np.abs(test_results[metric])
			if (not metric in avg_results): avg_results[metric] = data
			else: avg_results[metric] += data
	for metric in avg_results:
		avg_results[metric] /= num_tests
	print("{}".format(avg_results['esize']))

	return avg_results

# test_type: size, domain
def main():
	args = parse_args()
	test_type = args.test_type
	if (not os.path.exists(args.model_results_dir)): 
		raise Exception("Model {} doesn't exist".format(args.model_results_dir))
	# Collect results
	collected_results = dict() # "domain1" -> a data point/a list of data points
	versions = get_versions(test_type)
	valid_versions = []
	for version in versions:
		print(version)
		path = os.path.join(args.model_results_dir, version)
		if (not os.path.exists(path)): continue
		valid_versions.append(version)
		version_results = load_and_avg_over_tests(path=path)
		collected_results[version] = version_results["esize"]

	print("valid versions", valid_versions)
	print(collected_results)
	# print_all_results(collected_results, valid_versions, args)
	plot_std = test_type.startswith("allsize") or test_type.startswith("accdomain") or test_type.startswith("moredomain")
	plot_average(collected_results, valid_versions, args, plot_std=plot_std)


main()




