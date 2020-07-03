# standard library
from itertools import combinations
import numpy as np
import os, sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

np.random.seed(42)

words2 = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]

words3 = [
    ["jewish", "christian", "muslim"],
    ["jews", "christians", "muslims"],
    ["torah", "bible", "quran"],
    ["synagogue", "church", "mosque"],
    ["rabbi", "priest", "imam"],
    ["judaism", "christianity", "islam"],
]

DIRECTORY = '../text_corpus/'

GENDER = 0
RACE = 1

def match(a,L):
	for b in L:
		if a == b:
			return True
	return False

def replace(a,new,L):
	Lnew = []
	for b in L:
		if a == b:
			Lnew.append(new)
		else:
			Lnew.append(b)
	return ' '.join(Lnew)

def template2(words, sent, sent_list, all_pairs):
	for i, (female, male) in enumerate(words):
		if match(female, sent_list):
			sent_f = sent
			sent_m = replace(female,male,sent_list)
			all_pairs[i]['f'].append(sent_f)
			all_pairs[i]['m'].append(sent_m)
		if match(male, sent_list):
			sent_f = replace(male,female,sent_list)
			sent_m = sent
			all_pairs[i]['f'].append(sent_f)
			all_pairs[i]['m'].append(sent_m)
	return all_pairs

def template3(words, sent, sent_list, all_pairs):
	for (b1,b2,b3) in words:
		if match(b1, sent_list):
			sent_b1 = sent
			sent_b2 = replace(b1,b2,sent_list)
			sent_b3 = replace(b1,b3,sent_list)
			pair = (sent_b1,sent_b2,sent_b3)
			all_pairs.append(pair)
		if match(b2, sent_list):
			sent_b1 = replace(b2,b1,sent_list)
			sent_b2 = sent
			sent_b3 = replace(b2,b3,sent_list)
			pair = (sent_b1,sent_b2,sent_b3)
			all_pairs.append(pair)
		if match(b3, sent_list):
			sent_b1 = replace(b3,b1,sent_list)
			sent_b2 = replace(b3,b2,sent_list)
			sent_b3 = sent
			pair = (sent_b1,sent_b2,sent_b3)
			all_pairs.append(pair)
	return all_pairs

def get_pom():
	all_pairs2 = defaultdict(lambda: defaultdict(list))
	all_pairs3 = []
	pom_loc = os.path.join(DIRECTORY, 'POM/')
	total = 0
	num = 0
	for file in os.listdir(pom_loc):
		if file.endswith(".txt"):
			f = open(os.path.join(pom_loc, file), 'r')
			data = f.read()
			for sent in data.lower().split('.'):
				sent = sent.strip()
				sent_list = sent.split(' ')

				total += len(sent_list)
				num += 1
				all_pairs2 = template2(words2, sent, sent_list, all_pairs2)
				all_pairs3 = template3(words3, sent, sent_list, all_pairs3)
	return all_pairs2, all_pairs3

def get_rest(filename):
	all_pairs2 = defaultdict(lambda: defaultdict(list))
	all_pairs3 = []
	total = 0
	num = 0
	
	f = open(os.path.join(DIRECTORY, filename), 'r')
	data = f.read()
	for sent in data.lower().split('\n'):
		sent = sent.strip()
		sent_list = sent.split(' ')
		total += len(sent_list)
		num += 1
		all_pairs2 = template2(words2, sent, sent_list, all_pairs2)
		all_pairs3 = template3(words3, sent, sent_list, all_pairs3)

	print(filename, len(all_pairs2))
	return all_pairs2, all_pairs3

def get_sst():
	all_pairs2 = defaultdict(lambda: defaultdict(list))
	all_pairs3 = []
	total = 0
	num = 0
	for sent in open(os.path.join(DIRECTORY,'sst.txt'), 'r'):
		try:
			num = int(sent.split('\t')[0])
			sent = sent.split('\t')[1:]
			sent = ' '.join(sent)
		except:
			pass
		sent = sent.lower().strip()
		sent_list = sent.split(' ')
		total += len(sent_list)
		num += 1
		all_pairs2 = template2(words2, sent, sent_list, all_pairs2)
		all_pairs3 = template3(words3, sent, sent_list, all_pairs3)
	return all_pairs2, all_pairs3

def get_more_domains():
	def sample(data, n_samples=1000):
		n = len(data)
		indices = np.random.choice(n, n_samples, replace=False)
		sampled_data = []
		for index in indices: sampled_data.append(data[index])
		return sampled_data
	print("More domains")
	domains = ["reddit", "sst", "pom", "wikitext"]

	bucket_list = []
	for i, domain in enumerate(domains):
		domain_data = get_single_domain(domain)
		domain_data = sample(domain_data)
		print(domain, len(domain_data))
		if (i == 0): 
			bucket_list.append(domain_data)
		else: 
			bucket_list.append(bucket_list[-1] + domain_data)

	print("bucket sizes:")
	for bucket in bucket_list:
		print(len(bucket))
	return bucket_list

'''
Collect n_samples templates from each source: reddit, sst, pom, wikitext
Return a list where each element is a list of sentence pairs from one source.
'''
def get_all_domains(n_samples):
	def sample(data):
		n = len(data)
		indices = np.random.choice(n, n_samples, replace=False)
		sampled_data = []
		for index in indices: sampled_data.append(data[index])
		return sampled_data
	bucket_list = []
	domains = ["reddit", "sst", "pom", "wikitext"]
	for i, domain in enumerate(domains):
		domain_data = get_single_domain(domain)
		domain_data = sample(domain_data)
		bucket_list.append(domain_data)

	return bucket_list

def old_get_more_domains():
	print("More domains")
	b21, b31 = get_rest('news.txt')
	print ('news', len(b21), len(b31))
	b22, b32 = get_rest('reddit.txt')
	print ('reddit', len(b22), len(b32))
	b23, b33 = get_sst()
	print ('sst', len(b23), len(b33))
	b24, b34 = get_pom()
	print ('pom', len(b24), len(b34))
	b25, b35 = get_rest('wikitext.txt')
	print ('wikitext', len(b25), len(b35))
	
	b22 += b21
	b32 += b31
	b23 += b22
	b33 += b32
	b24 += b23
	b34 += b33
	b25 += b24
	b35 += b34

	bucket1 = (b21,b31)
	bucket2 = (b22,b32)
	bucket3 = (b23,b33)
	bucket4 = (b24,b34)
	bucket5 = (b25,b35)

	print(len(b21), len(b31))
	print(len(b22), len(b32))
	print(len(b23), len(b33))
	print(len(b24), len(b34))
	print(len(b25), len(b35))
	return bucket1, bucket2, bucket3, bucket4, bucket5


def get_single_domain_in_buckets(domain="wikitext", buckets=5):
	print("Same domain with divided into buckets")
	all_pairs_gender = get_single_domain(domain)
	
	bucket_size = int(len(all_pairs_gender)/buckets)
	bucket_list = []
	for i in range(buckets):
		bucket_list.append(all_pairs_gender[i*bucket_size:(i+1)*bucket_size])

	return bucket_list

def get_same_domain_more_size(domain="wikitext"):
	print("Same domain with different sizes")
	all_pairs2, all_pairs3 = get_single_domain(domain)
	print (domain, len(all_pairs2), len(all_pairs3))

	buckets = 5
	each2 = int(len(all_pairs2)/5)
	each3 = int(len(all_pairs3)/5)
	b21 = all_pairs2[:each2]
	b22 = all_pairs2[:2*each2]
	b23 = all_pairs2[:3*each2]
	b24 = all_pairs2[:4*each2]
	b25 = all_pairs2

	b31 = all_pairs3[:each3]
	b32 = all_pairs3[:2*each3]
	b33 = all_pairs3[:3*each3]
	b34 = all_pairs3[:4*each3]
	b35 = all_pairs3

	bucket1 = (b21,b31)
	bucket2 = (b22,b32)
	bucket3 = (b23,b33)
	bucket4 = (b24,b34)
	bucket5 = (b25,b35)

	print(len(b21), len(b31))
	print(len(b22), len(b32))
	print(len(b23), len(b33))
	print(len(b24), len(b34))
	print(len(b25), len(b35))
	return bucket1, bucket2, bucket3, bucket4, bucket5

def check_bucket_size(D):
	n = 0
	for i in D:
		for key in D[i]:
			n += len(D[i][key])
			break
	return n

# domain: news, reddit, sst, pom, wikitext
def get_single_domain(domain):
	if (domain == "pom"):
		gender, race = get_pom()
	elif (domain == "sst"):
		gender, race = get_sst()
	else:
		gender, race = get_rest("{}.txt".format(domain))
	return gender

def get_all():
	domains = ["reddit", "sst", "wikitext", "pom", "meld", "news_200"] #, "yelp_review_10mb"] # "news_200"]
	print("Get data from {}".format(domains))
	all_data = defaultdict(lambda: defaultdict(list))
	for domain in domains:
		bucket = get_single_domain(domain)
		bucket_size = check_bucket_size(bucket)
		print("{} has {} pairs of templates".format(domain, bucket_size))
		for i in bucket:
			for term in bucket[i]:
				all_data[i][term].extend(bucket[i][term])
	total_size = check_bucket_size(all_data)
	print("{} pairs of templates in total".format(total_size))
	return all_data

def get_def_pairs(def_pairs_name):
	eqsize_prefix = 'eqsize'
	# all 5 sets
	if (def_pairs_name == "all"):
		return get_all()
	elif (def_pairs_name.startswith(eqsize_prefix)):
		n_samples = int(def_pairs_name[len(eqsize_prefix):])
		print("Select {} templates from each source.".format(n_samples))
		domain_list = get_all_domains(n_samples)
		def_pairs = []
		for domain_data in domain_list: def_pairs.extend(domain_data)
		return def_pairs
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
	data = get_all()