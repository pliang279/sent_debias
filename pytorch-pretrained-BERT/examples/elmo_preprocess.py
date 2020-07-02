from __future__ import absolute_import, division, print_function
import numpy as np 
import argparse
import os, sys, math, time
from allennlp.commands.elmo import ElmoEmbedder
import csv
from tqdm import tqdm

'''
python elmo_preprocess.py --set train --data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/SST-2 --task_name sst-2
python elmo_preprocess.py --set train --data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/CoLA --task_name CoLA
'''

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				lines.append(line)
			return lines


class Sst2Processor(DataProcessor):
	"""Processor for the SST-2 data set (GLUE version)."""

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		label_list = self.get_labels()
		label_map = {label : i for i, label in enumerate(label_list)} 
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = line[0]
			label = line[1]
			label_id = label_map[label]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label_id))
		return examples


class ColaProcessor(DataProcessor):
	"""Processor for the CoLA data set (GLUE version)."""

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		label_list = self.get_labels()
		label_map = {label : i for i, label in enumerate(label_list)} 
		examples = []
		for (i, line) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text_a = line[3]
			label = line[1]
			label_id = label_map[label]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label_id))
		return examples


processors = {
	"cola": ColaProcessor,
	"sst-2": Sst2Processor
}

class ElmoEncoder(object):

	def __init__(self):
		self.elmo = ElmoEmbedder()

	# return: numpy array
	def encode_batch(self, sents):
		vec_seq = self.elmo.embed_sentences(sents)
		vecs = []
		for vec in vec_seq:
			vecs.append(self.collapse_vec(vec))
		# vecs = torch.stack(vecs)
		vecs = np.stack(vecs)
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


def parse_args():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--task_name",
						default=None,
						type=str,
						required=True,
						help="The name of the task to train.")
	parser.add_argument("--set",
						default=None,
						type=str,
						choices = ["train", "dev"],
						required=True,
						help="train or eval")
	parser.add_argument("--output_dir",
						default="elmo_data",
						type=str,
						help="The output directory where the embeddings will be stored.")

	args = parser.parse_args()
	args.task_name = args.task_name.lower()

	return args




def get_embeddings(examples, encoder, label_list):
	label_map = {label : i for i, label in enumerate(label_list)} 
	def collate_batch(batch_examples):
		# start_time = time.time()
		texts = []
		labels = []
		for example in batch_examples:
			texts.append(example.text_a)
			labels.append(example.label)

		# elapsed_time = time.time() - start_time
		# print("collate time={}".format(elapsed_time))
		return texts, labels
	batch_size = 32
	embeddings = []
	label_list = []

	nexamples = len(examples)
	num_batch = int(math.floor(nexamples / batch_size))
	for i in tqdm(range(num_batch), desc="Encoding"):
		batch_examples = examples[i*batch_size:(i+1)*batch_size]
		texts, batch_labels = collate_batch(batch_examples)
		# start_time = time.time()
		batch_ebds = encoder.encode_batch(texts)
		# elapsed_time = time.time() - start_time
		# print("encode time={}".format(elapsed_time))

		batch_ebds = np.array(batch_ebds)
		batch_labels = np.array(batch_labels)
		embeddings.append(batch_ebds)
		label_list.append(batch_labels)

	# flatten
	embeddings = np.concatenate(embeddings)
	label_list = np.concatenate(label_list)
	print("embeddings={} labels={}".format(embeddings.shape, label_list.shape))
	# zip
	data = list(zip(embeddings, label_list))
	data = np.array(data)
	return data


def main():

	args = parse_args()
	
	processor = processors[args.task_name]()
	label_list = processor.get_labels()
	if (args.set == "train"):
		examples = processor.get_train_examples(args.data_dir)
	elif (args.set == "dev"):
		examples = processor.get_dev_examples(args.data_dir)
	else:
		raise NotImplementedError
	print("examples={}".format(len(examples)))

	elmo_encoder = ElmoEncoder()

	start_time = time.time()
	data = get_embeddings(examples, elmo_encoder, label_list)
	elapsed_time = time.time() - start_time
	print("Time={}".format(elapsed_time))

	if (not os.path.exists(args.output_dir)): os.makedirs(args.output_dir)
	output_file = os.path.join(args.output_dir, "{};{}".format(args.task_name, args.set))

	np.save(output_file, data)

if __name__ == '__main__':
	main()

