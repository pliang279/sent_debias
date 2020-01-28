# standard library
from itertools import combinations
import numpy as np

# first party
from bias_data.def_sent_pairs import pairs_dict
from bias_data.diversity_test import *

GENDER = 0
RACE = 1

def get_def_pairs(def_pairs_name):
	# old sets
	if (def_pairs_name == "all"):
		return get_all()
	elif (def_pairs_name in pairs_dict):
		return pairs_dict[def_pairs_name]
	# wikitext, with varying size
	elif (def_pairs_name.startswith("size")):
		size = int(def_pairs_name[4:])
		buckets = get_same_domain_more_size()
		bucket = buckets[size-1][GENDER]
		return bucket
	# varying number of domains
	elif (def_pairs_name.startswith("domain")):
		num_domains = int(def_pairs_name[6:])
		buckets = old_get_more_domains()
		bucket = buckets[num_domains-1][GENDER]
		return bucket
	# accumulating domains with same number of samples
	elif (def_pairs_name.startswith("accdomain")):
		start_idx = len("accdomain")
		num_domains = int(def_pairs_name[start_idx:])
		buckets = get_more_domains()
		bucket = buckets[num_domains-1]
		return bucket
	# single-domain
	elif (def_pairs_name in ["news", "reddit", "sst", "pom", "wikitext"]):
		return get_single_domain(def_pairs_name)
	else:
		raise Exception("invalid defining pairs name")

if __name__ == '__main__':
	for domain in ["news", "reddit", "sst", "pom", "wikitext"]:
		get_def_pairs(domain)
	for i in range(1, 6):
		get_def_pairs("size{}".format(i))
		get_def_pairs("domain{}".format(i))

