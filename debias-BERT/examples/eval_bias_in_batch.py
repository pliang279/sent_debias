# standard library
import os
import numpy as np
import argparse

# example usage:
# python eval_bias_in_batch.py -m biased_sst-2 -d 0

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m",
					type=str,
					required=True,
					help="The name of the model to evaluate.")
parser.add_argument("--device", "-d",
					type=int,
					required=True,
					help="GPU name.")
args = parser.parse_args()

name_to_path = {"biased_sst-2": "aaai-results/sst-2/biased/", 
		"biased_qnli": "aaai-results/qnli/biased/",
		"biased_cola": "aaai-results/cola/old_biased",
		"pretrained": None}

def test(test_type, device, test_id=None):
	model_name = args.model
	model_path = name_to_path[model_name]
	assert(test_type in ["size", "domain", "allsize", "accdomain", "moredomain"])
	if (test_id == None): ids = np.arange(5)
	else: ids = [test_id]
	for i in ids:
		cmd = "CUDA_VISIBLE_DEVICES={} python eval_bias.py".format(device)
		if(i > 0): 
			def_pairs_name = "{}{}".format(test_type, i)
			output_name = def_pairs_name
			cmd += " --debias"
			cmd += " --def_pairs_name {}".format(def_pairs_name)
		else:
			output_name = "biased"
		cmd += " --output_name {}".format(output_name)
		cmd += " --model {}".format(model_name)
		if (model_path):
			cmd += " --model_path {}".format(model_path)
		print(cmd)
		os.system(cmd)

def test_single_domain(device):
	model_name = args.model
	model_path = name_to_path[model_name]
	def_pairs_names = [None, "news", "reddit", "sst", "pom", "wikitext"]
	for def_pairs_name in def_pairs_names:
		cmd = "CUDA_VISIBLE_DEVICES={} python eval_bias.py".format(device)
		if(def_pairs_name): 
			output_name = def_pairs_name
			cmd += " --debias"
			cmd += " --def_pairs_name {}".format(def_pairs_name)
		else:
			output_name = "biased"
		cmd += " --output_name {}".format(output_name)
		cmd += " --model {}".format(model_name)
		if (model_path):
			cmd += " --model_path {}".format(model_path)
		os.system(cmd)

if __name__ == '__main__':
	# test("moredomain", device=args.device)
	# test("allsize", device=args.device)
	test("domain", 2)
	# test("size", 2)
