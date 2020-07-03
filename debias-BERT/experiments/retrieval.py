import pickle
import os
import collections

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# retrieval on biased/debiased sentence embeddings
# model is pretrained BERT

DATA_DIR = "./acl_bias_eval_results/pretrained"

def load_encs(test_id, debiased):
	bias_flag = "debiased" if debiased else "biased"
	filename = "{}_encs{}.pkl".format(bias_flag, test_id)
	with open(os.path.join(DATA_DIR, filename), 'rb') as f:
		encs = pickle.load(f)

	# return encs
	# Levels of keys in encs:
	# 1: 'targ1', 'targ2', 'attr1', 'attr2'
	# 2: 'category', 'examples', 'text_ids', 'encs'
	# 3: 'encs': text->embedding
	gender_data = dict()
	nogender_data = dict()
	group_names = encs.keys()
	for group_name in group_names:
		group = encs[group_name]
		category = group["category"]
		group_encs = group["encs"]
		text_ids = group["text_ids"]
		group_size = len(text_ids)
		# print("group={} category={} size={}".format(group_name, category, group_size))
		# print(text_ids.values())
		text2encs = dict()
		for text_id in text_ids:
			text = text_ids[text_id]
			sent_enc = group_encs[str(text_id)]
			text2encs[text] = sent_enc
		if ("male" in category.lower()):	
			gender_data[category] = text2encs
		else:
			nogender_data[category] = text2encs
	return gender_data, nogender_data

def index_list(A, indices):
	return [A[i] for i in indices]

def get_source_sents(gender_data, avg):
	if (not avg):
		# choose gender specific sentences
		gender_key = list(gender_data.keys())[0].lower()
		if ("term" in gender_key):
			# source_sents = ["Here he is.", "There she is."]
			source_sents = ["This is a man.", "This is a woman."]
		elif ("name" in gender_key):
			source_sents = ["This is John.", "This is Amy."]

		source_dict = dict()
		# obtain source embeddings
		for category in gender_data:
			text2encs = gender_data[category]
			# print(text2encs.keys())
			for sent in source_sents:
				if (sent in text2encs):
					source_dict[sent] = text2encs[sent]
		for sent in source_sents: assert(sent in source_dict)
	else:
		source_dict = dict()
		for category in gender_data:
			category_data = gender_data[category]
			avg_ebd = np.array([category_data[text] for text in category_data])
			avg_ebd = np.mean(avg_ebd, axis=0)
			source_dict[category] = avg_ebd
	return source_dict

def retrieve_topk(test_id, debiased, k=10):
	gender_data, nogender_data = load_encs(test_id, debiased)
	if (debiased): print("DEBIASED")
	else: print("BIASED")
	# print("gendered categories: {}".format(gender_data.keys()))
	# print("non-gendered categories: {}".format(nogender_data.keys()))
	
	source_dict = get_source_sents(gender_data, avg=True)
	sent_keys = list(source_dict.keys())
	sources = np.array([source_dict[sent] for sent in sent_keys])

	# construct targets
	targets = []
	cat_assignment = []
	target_texts = []
	for category in nogender_data:
		text2encs = nogender_data[category]
		for text in text2encs:
			targets.append(text2encs[text])
			cat_assignment.append(category)
			target_texts.append(text)
	targets = np.array(targets)
	print("sources={} targets={}".format(sources.shape, targets.shape))

	# start retrieving!
	sim_scores = -cosine_similarity(sources, targets)
	rank_matrix = np.argsort(sim_scores, axis=-1)
	for i, source_sent in enumerate(sent_keys):
		count_dict = collections.defaultdict(int)
		print("#"*80)
		print("source={}".format(source_sent))
		top_indices = rank_matrix[i,:k]
		for rank in range(k):
			index = top_indices[rank]
			category = cat_assignment[index]
			count_dict[category] += 1
			print(category, target_texts[index], sim_scores[i,index])
		print(count_dict)

test_id = "7b"
retrieve_topk(test_id=test_id, debiased=False)
retrieve_topk(test_id=test_id, debiased=True)

'''
6: family/career
7:3, 9:1 -> 7:3, 8:2
'''















