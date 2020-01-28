import numpy as np
import os
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

DIRECTORY = '../bias_data'
# DIRECTORY = "/media/bighdd7/irene/debias/sent_tasks/pytorch-pretrained-BERT/bias_data"
# DIRECTORY = '/work/mengzeli/debias/sent_tasks/pytorch-pretrained-BERT/bias_data/'

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
	for (female, male) in words:
		if match(female, sent_list):
			sent_f = sent
			sent_m = replace(female,male,sent_list)
			pair = (sent_f,sent_m)
			all_pairs.append(pair)
		if match(male, sent_list):
			sent_f = replace(male,female,sent_list)
			sent_m = sent
			pair = (sent_f,sent_m)
			all_pairs.append(pair)
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
	all_pairs2 = []
	all_pairs3 = []
	pom_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/POM/Raw/Transcript/Full'
	total = 0
	num = 0
	for file in os.listdir(pom_loc):
		if file.endswith(".txt"):
			f = open(os.path.join(pom_loc, file), 'r')
			data = f.read()
			for sent in data.lower().split('.'):
				sent = sent.strip()
				sent_list = sent.split(' ')
				if True: # len(sent_list) < 10:
					total += len(sent_list)
					num += 1
					all_pairs2 = template2(words2, sent, sent_list, all_pairs2)
					all_pairs3 = template3(words3, sent, sent_list, all_pairs3)
	# print (all_pairs2, len(all_pairs2)) # total/float(num))
	# print (all_pairs3, len(all_pairs3)) # total/float(num))
	return all_pairs2, all_pairs3

def get_rest(filename):
	all_pairs2 = []
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

	# print (all_pairs2, len(all_pairs2)) # total/float(num))
	# print (all_pairs3, len(all_pairs3)) # total/float(num))
	return all_pairs2, all_pairs3

def get_sst():
	all_pairs2 = []
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
	# print (all_pairs2, len(all_pairs2)) # total/float(num))
	# print (all_pairs3, len(all_pairs3)) # total/float(num))
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
		else: bucket_list.append(bucket_list[-1] + domain_data)

	print("bucket sizes:")
	for bucket in bucket_list:
		print(len(bucket))
	return bucket_list

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

# domain: news, reddit, sst, pom, wikitext
def get_single_domain(domain):
	if (domain == "pom"):
		gender, race = get_pom()
	elif (domain == "sst"):
		gender, race = get_sst()
	else:
		gender, race = get_rest("{}.txt".format(domain))
	return gender

def test_get_single_domain():
	for domain in ["news", "reddit", "sst", "pom", "wikitext"]:
		bucket = get_single_domain(domain)
		print(domain, len(bucket))

def get_all():
	domains = ["news", "reddit", "sst", "wikitext"]
	print("Get data from {}".format(domains))
	buckets = []
	for domain in domains:
		bucket = get_single_domain(domain)

		buckets.extend(bucket)
	for i in range(5):
		print(i, buckets[i])
	return buckets

if __name__ == '__main__':
	# get_more_domains()
	# old_get_more_domains()
	# get_same_domain_more_size()
	buckets = get_all()










