import os
import argparse
import json, pickle
from collections import defaultdict

from pattern.en import pluralize, singularize
import nltk
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import gensim.downloader as api

from run_classifier import get_encodings, compute_gender_dir, get_tokenizer_encoder

DATA_DIR = "../bias_data/gender_tests/"

def load_json(file):
	''' Load from json. We expect a certain format later, so do some post processing '''
	all_data = json.load(open(file, 'r'))
	data = {}
	for k, v in all_data.items():
		examples = v["examples"]
		data[k] = examples
	return all_data  # data

def my_pluralize(word):
	if (word in ["he", "she", "her", "hers"]): return word
	if (word == "brother"): return "brothers"
	if (word == "drama"): return "dramas"
	return pluralize(word)

def my_singularize(word):
	if (word in ["hers", "his", "theirs"]): return word
	return singularize(word)

def match_one_test(test_name):
	# load words
	word_filename = "weat{}.jsonl".format(test_name)
	word_file = os.path.join(DATA_DIR, word_filename)
	word_data = load_json(word_file)

	# load simple sentences
	sent_filename = "sent-weat{}.jsonl".format(test_name)
	sent_file = os.path.join(DATA_DIR, sent_filename)
	sent_data = load_json(sent_file)

	word2sents = dict()
	num_sents = 0
	for key in ['targ1', 'targ2', 'attr1', 'attr2']:
		words = word_data[key]['examples']
		for word in words: word2sents[word] = []
	all_words = set(word2sents.keys())
	print("all words")
	print(all_words)
	unmatched_sents = []
	for key in ['targ1', 'targ2', 'attr1', 'attr2']:
		sents = sent_data[key]['examples']
		for sent in sents:
			matched = False
			for word in all_words:
				word_ = word.lower()
				sent_ = sent.lower()
				tokens = nltk.word_tokenize(sent_)
				word_variants = set({word})
				word_variants.add(my_pluralize(word_))
				word_variants.add(my_singularize(word_))
				matched_words = []
				for word_variant in word_variants:
					if (word_variant in tokens):
						matched_words.append(word)
						if (matched): 
							print("'{}' is matched to {}!.".format(sent, word))
							print(matched_words)
						matched = True
						word2sents[word].append(sent)
						break
			if (not matched): unmatched_sents.append(sent)
	with open(os.path.join(DATA_DIR, 'word2sents{}.jsonl'.format(test_name)), 'w') as outfile:
		json.dump(word2sents, outfile)
	print("unmatched: {}".format(unmatched_sents))
	return

def match():
	for test_name in [6, 7, 8]:
		match_one_test("{}".format(test_name))
		match_one_test("{}b".format(test_name))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path",
						type=str,
						default="bert-base-uncased",
						help="Path of the model to be evaluated")
	parser.add_argument("--debias",
						action='store_true',
						help="Whether to debias.")
	parser.add_argument("--equalize",
						action='store_true',
						help="Whether to equalize.")
	parser.add_argument("--def_pairs_name", default="large_real", type=str,
						help="Name of definitional sentence pairs.")
	parser.add_argument("--model", "-m", type=str, default="dummy")
	parser.add_argument("--output_name", type=str, required=True)
	args = parser.parse_args()
	args.results_dir = os.path.join("acl_bias_eval_results", args.model)
	args.do_lower_case = True
	args.cache_dir = None
	args.local_rank = -1
	args.max_seq_length = 128
	args.eval_batch_size = 8
	args.n_samples = 100000
	args.parametric = True
	args.tune_bert = False
	args.normalize = True

	# word embeddings
	args.word_model = 'fasttext-wiki-news-subwords-300'
	wedata_path = 'my_debiaswe/data'
	args.definitional_filename = os.path.join(wedata_path, 'definitional_pairs.json')
	args.equalize_filename = os.path.join(wedata_path, 'equalize_pairs.json')
	args.gendered_words_filename = os.path.join(wedata_path, 'gender_specific_full.json')

	return args

def get_sent_vectors(test_name, debiased):
	if (debiased):
		sent_encs_filename = "debiased_sent_encs{}.pkl".format(test_name)
	else:
		sent_encs_filename = "sent_encs{}.pkl".format(test_name)
	file = open(os.path.join(DATA_DIR, sent_encs_filename), 'rb')
	data = pickle.load(file)

	all_sent_vectors = dict()
	for key in ['targ1', 'targ2', 'attr1', 'attr2']:
		text_ids = data[key]['text_ids']
		encs = data[key]['encs']
		for i in text_ids:
			text = text_ids[i]
			vector = encs[str(i)]
			all_sent_vectors[text] = vector
	print("loaded sentence vectors")
	return all_sent_vectors

def tsne_plot(word_vectors, word_labels, plot_name, title, do_PCA=False):
	words = list(word_vectors.keys())
	X = np.array([word_vectors[word] for word in words])

	# PCA (optional)
	if (do_PCA):
		print("PCA")
		pca = PCA()
		pca.fit(X.T)
		components = pca.components_
		X = components.T # Nx50
		print("After PCS: {}".format(X.shape))

	# t-SNE
	X_embedded = TSNE(n_components=2, perplexity=len(X)-1).fit_transform(X) # Nx2
	y, z = X_embedded[:, 0], X_embedded[:, 1]

	data_by_label = defaultdict(list)
	for i, word in enumerate(words):
		label = word_labels[word]
		data_by_label[label].append(i)

	fig, ax = plt.subplots()
	colors = ['r', 'g', 'b', 'c']
	for label_id, label in enumerate(data_by_label.keys()):
		indices = np.array(data_by_label[label])
		sub_y = y[indices]
		sub_z = z[indices]
		ax.scatter(sub_y, sub_z, label=label, c=colors[label_id])
		for word_id in indices:
			ax.annotate(words[word_id], (y[word_id], z[word_id]), size=14)
	ax.legend()
	directory = "tsne_plots"
	if (not os.path.exists(directory)): os.makedirs(directory)
	ax.title(title)
	plt.savefig(os.path.join(directory, plot_name))
	plt.clf()

def simple_tsne_plot(word_vectors, perplexity, title, filename, do_PCA=True, do_tsne=False):
	assert(do_PCA or do_tsne)
	if (do_tsne): assert(perplexity != None)
	words = ['woman', 'man', 'family', 'career', 'math', 'art', 
		"science", "literature", "technology", "dance"]
	X = np.array([word_vectors[word] for word in words])

	# PCA (optional)
	if (do_PCA):
		print("PCA")
		pca = PCA()
		pca.fit(X.T)
		components = pca.components_
		X = components.T # Nx50
		print("After PCS: {}".format(X.shape))

	# t-SNE
	if (do_tsne):
		X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(X) # Nx2
		X1, X2 = X_embedded[:, 0], X_embedded[:, 1]
	else:
		X1, X2 = X[:, 0], X[:, 1]

	word_dict = dict()
	for i, word in enumerate(words):
		word_dict[word] = (X1[i], X2[i])
	with open('{}.pkl'.format(filename), 'wb') as f:
		pickle.dump(word_dict, f)
	print("write to {}".format(filename))
	return

	color_dict = {"woman": 'r', "man": "b"}
	colors = [color_dict.get(word, 'k') for word in words]
	plt.scatter(X1, X2, color=colors)
	for word_id, word in enumerate(words):
		plt.annotate(word, (X1[word_id], X2[word_id]))
	x_margin = (max(X1)-min(X1)) * 0.1
	y_margin = (max(X2)-min(X2)) * 0.1
	plt.xlim(min(X1)-x_margin, max(X1)+x_margin)
	plt.ylim(min(X2)-y_margin, max(X2)+y_margin)
	plt.title(title, fontsize=20)
	plt.xticks([])
	plt.yticks([])
	plot_dir = "visual_plots"
	filename = os.path.join(plot_dir, plot_name)
	directory = os.path.dirname(filename)
	if (not os.path.exists(directory)): os.makedirs(directory)
	print("Saving to {}".format(filename))
	plt.savefig(filename)
	plt.clf()


def get_word_labels(test_name):
	word_filename = "weat{}.jsonl".format(test_name)
	word_file = os.path.join(DATA_DIR, word_filename)
	word_data = load_json(word_file)

	labels = dict()
	for key in ['targ1', 'targ2', 'attr1', 'attr2']:
		words = word_data[key]['examples']
		category = word_data[key]['category']
		for word in words: labels[word] = category
	return labels

def visualize_test(test_name, debiased):
	bias_flag = "debiased" if debiased else "biased"
	print("Visualize {} {}".format(bias_flag, test_name))
	file_name = os.path.join(DATA_DIR, 'word2sents{}.jsonl'.format(test_name))
	word2sents = json.load(open(file_name, 'r'))
	print(list(word2sents.keys()))

	all_sent_vectors = get_sent_vectors(test_name, debiased)
	word_labels = get_word_labels(test_name)

	word_vectors = dict()
	for word in word2sents:
		sents = word2sents[word]
		sent_vectors = np.array([all_sent_vectors[sent] for sent in sents])
		word_vector = np.mean(sent_vectors, axis=0)
		word_vectors[word] = word_vector

	tsne_plot(word_vectors, word_labels, "{}_{}.png".format(bias_flag, test_name),
		"Test {} {} Word Embedding t-SNE Plot".format(test_name, bias_flag))

def visualize_all(debiased):
	for test_name in ['6', '6b', '7', '7b', '8', '8b']:
		visualize_test(test_name, debiased=debiased)

def words_from_sents(word2sents, test_name, debiased):
	all_sent_vectors = get_sent_vectors(test_name, debiased)

	word_vectors = dict()
	for word in word2sents:
		sents = word2sents[word]
		sent_vectors = np.array([all_sent_vectors[sent] for sent in sents])
		sent_vectors = sent_vectors / LA.norm(sent_vectors, ord=2, axis=-1, keepdims=True)
		word_vector = np.mean(sent_vectors, axis=0)
		word_vector = word_vector / LA.norm(word_vector, ord=2, axis=-1, keepdims=True)
		# word_vector = all_sent_vectors[sents[0]]
		word_vectors[word] = word_vector
	return word_vectors

def get_fasttext(words):
	model = api.load("fasttext-wiki-news-subwords-300") # takes a few minutes
	word_vectors = dict()
	for word in words:
		word_vectors[word] = model.word_vec(word)
	return word_vectors

def visualize_few_words(debiased, do_PCA, do_tsne, perplexity=None, use_sents=True):
	assert(do_PCA or do_tsne)
	bias_flag = "debiased" if debiased else "biased"
	sent_flag = "sent" if use_sents else "word"
	print("Visualize a few words")
	all_word_vectors = dict()
	for test_name in ['6', '7', '8']:
		file_name = os.path.join(DATA_DIR, 'word2sents{}.jsonl'.format(test_name))
		word2sents = json.load(open(file_name, 'r'))
		words = list(word2sents.keys())
		print(words)

		if (use_sents):
			word_vectors = words_from_sents(word2sents, test_name, debiased)
		else:
			word_vectors = get_fasttext(words)
		all_word_vectors.update(word_vectors)

	# Plot
	if (not do_tsne):
		directory = "pca"
		filename = "{}_{}_pca".format(bias_flag, sent_flag)
		# plot_name = "{}.png".format(filename)
		title = "{} Word Embedding PCA Plot".format(bias_flag.capitalize())
	elif (do_tsne and not do_PCA):
		directory = "tsne"
		filename = "{}_{}_p{}".format(bias_flag, sent_flag, perplexity)
		title = "{} Word Embedding t-SNE Plot".format(bias_flag.capitalize())
	elif (do_tsne and do_PCA):
		directory = "pca_tsne"
		filename = "{}_{}_pca_p{}".format(bias_flag, sent_flag, perplexity)
		title = "{} Word Embedding t-SNE Plot (perplexity={}) with PCA".format(bias_flag.capitalize(), perplexity)
	simple_tsne_plot(all_word_vectors, perplexity, title, 
		filename, do_PCA=do_PCA, do_tsne=do_tsne)	

if __name__ == '__main__':

	# tsne only
	for p in [4]:
		visualize_few_words(debiased=True, do_PCA=False, do_tsne=True, perplexity=p)
		visualize_few_words(debiased=False, do_PCA=False, do_tsne=True, perplexity=p)
	
	# PCA and tsne
	for p in [2, 4, 8, 16]:
		visualize_few_words(debiased=True, do_PCA=True, do_tsne=True, perplexity=p)
		visualize_few_words(debiased=False, do_PCA=True, do_tsne=True, perplexity=p)
	
	# PCA only
	visualize_few_words(debiased=True, do_PCA=True, do_tsne=False)
	visualize_few_words(debiased=False, do_PCA=True, do_tsne=False)