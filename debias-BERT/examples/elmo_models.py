from __future__ import print_function, division
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import time
import torch

class ElmoEncoder(object):

	def __init__(self):
		self.elmo = ElmoEmbedder()

	def encode_batch(self, sents):
		start_time = time.time()
		vec_seq = self.elmo.embed_sentences(sents)
		elapsed_time = time.time() - start_time
		print("embed_sentences {}".format(elapsed_time))
		vecs = []
		start_time = time.time()
		for vec in vec_seq:
			vecs.append(self.collapse_vec(vec))
		# vecs = torch.stack(vecs)
		vecs = np.stack(vecs)
		elapsed_time =time.time() - start_time
		print("collapse {}".format(elapsed_time))
		print("vecs ", vecs.shape)
		return vecs

	def collapse_vec(self, vec_seq, time_combine_method="max", layer_combine_method="add"):
		if time_combine_method == "max":
			vec = vec_seq.max(axis=1)
		elif time_combine_method == "mean":
			vec = vec_seq.mean(axis=1)
		elif time_combine_method == "concat":
			vec = np.concatenate(vec_seq, axis=1)
		elif time_combine_method == "last":
			vec = vec_seq[:, -1]
		else:
			raise NotImplementedError

		if layer_combine_method == "add":
			vec = vec.sum(axis=0)
		elif layer_combine_method == "mean":
			vec = vec.mean(axis=0)
		elif layer_combine_method == "concat":
			vec = np.concatenate(vec, axis=0)
		elif layer_combine_method == "last":
			vec = vec[-1]
		else:
			raise NotImplementedError

		return vec

	def encode(self, sents, time_combine_method="max", layer_combine_method="add"):
		""" Load ELMo and encode sents """
		vecs = {}
		for sent in sents:
			vec_seq = self.elmo.embed_sentence(sent)
			if time_combine_method == "max":
				vec = vec_seq.max(axis=1)
			elif time_combine_method == "mean":
				vec = vec_seq.mean(axis=1)
			elif time_combine_method == "concat":
				vec = np.concatenate(vec_seq, axis=1)
			elif time_combine_method == "last":
				vec = vec_seq[:, -1]
			else:
				raise NotImplementedError

			if layer_combine_method == "add":
				vec = vec.sum(axis=0)
			elif layer_combine_method == "mean":
				vec = vec.mean(axis=0)
			elif layer_combine_method == "concat":
				vec = np.concatenate(vec, axis=0)
			elif layer_combine_method == "last":
				vec = vec[-1]
			else:
				raise NotImplementedError
			vecs[' '.join(sent)] = vec
		return vecs


def encode(sents, time_combine_method="max", layer_combine_method="add"):
	""" Load ELMo and encode sents """
	elmo = ElmoEmbedder()
	vecs = {}
	for sent in sents:
		vec_seq = elmo.embed_sentence(sent)
		if time_combine_method == "max":
			vec = vec_seq.max(axis=1)
		elif time_combine_method == "mean":
			vec = vec_seq.mean(axis=1)
		elif time_combine_method == "concat":
			vec = np.concatenate(vec_seq, axis=1)
		elif time_combine_method == "last":
			vec = vec_seq[:, -1]
		else:
			raise NotImplementedError

		if layer_combine_method == "add":
			vec = vec.sum(axis=0)
		elif layer_combine_method == "mean":
			vec = vec.mean(axis=0)
		elif layer_combine_method == "concat":
			vec = np.concatenate(vec, axis=0)
		elif layer_combine_method == "last":
			vec = vec[-1]
		else:
			raise NotImplementedError
		vecs[' '.join(sent)] = vec # 1024
	return vecs

sentences = [
"hello, world!", 
"happy birthday!", 
"let's get it.",
"She's a girl.",
"That's a baby", 
"how long was it?"]
for _ in range(3):
	sentences += sentences
print(len(sentences))

start_time = time.time()
elmo_encoder = ElmoEncoder()
elapsed_time = time.time() - start_time
print("Initializing takes {}".format(elapsed_time))
start_time = time.time()
elmo_encoder.encode_batch(sentences)
elapsed_time = time.time() - start_time
print("Encoding takes {}".format(elapsed_time))
for sent in sentences:
	start_time = time.time()
	singleton_list = [sent]
	elmo_encoder.encode(singleton_list)
	elapsed_time = time.time() - start_time
	print(elapsed_time)

