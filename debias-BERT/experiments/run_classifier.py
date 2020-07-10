# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function

# standard library 
import argparse
import csv
import logging
import os
import random
import sys
import pickle
import pdb

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.decomposition import PCA

# first party
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from def_sent_utils import get_def_pairs
from eval_utils import isInSet
from my_debiaswe import my_we

logger = logging.getLogger(__name__)

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


class DualInputFeatures(object):
	"""A single set of dual features of data."""

	def __init__(self, input_ids_a, input_ids_b, mask_a, mask_b, segments_a, segments_b):
		self.input_ids_a = input_ids_a
		self.input_ids_b = input_ids_b
		self.mask_a = mask_a
		self.mask_b = mask_b
		self.segments_a = segments_a
		self.segments_b = segments_b


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, tokens, input_ids, input_mask, segment_ids, label_id):
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


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
		examples = []
		for (i, line) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text_a = line[3]
			label = line[1]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples


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
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = line[0]
			label = line[1]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples


class QnliProcessor(DataProcessor):
	"""Processor for the STS-B data set (GLUE version)."""

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
			"dev_matched")

	def get_labels(self):
		"""See base class."""
		return ["entailment", "not_entailment"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, line[0])
			text_a = line[1]
			text_b = line[2]
			label = line[-1]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


class BertEncoder(object):
	def __init__(self, model, device):
		self.device = device
		self.bert = model

	def encode(self, input_ids, token_type_ids=None, attention_mask=None, word_level=False):
		self.bert.eval()
		embeddings = self.bert(input_ids, token_type_ids=token_type_ids, 
			attention_mask=attention_mask, word_level=word_level, 
			remove_bias=False, bias_dir=None, encode_only=True)
		return embeddings


def extract_embeddings(bert_encoder, tokenizer, examples, max_seq_length, device, 
		label_list, output_mode, norm, word_level=False):
	'''Encode examples into BERT embeddings in batches.'''
	features = convert_examples_to_dualfeatures(
		examples, label_list, max_seq_length, tokenizer, output_mode)
	all_inputs_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
	all_mask_a = torch.tensor([f.mask_a for f in features], dtype=torch.long)
	all_segments_a = torch.tensor([f.segments_a for f in features], dtype=torch.long)

	data = TensorDataset(all_inputs_a, all_mask_a, all_segments_a)
	dataloader = DataLoader(data, batch_size=32, shuffle=False)
	all_embeddings = []
	for step, batch in enumerate(tqdm(dataloader)):
		inputs_a, mask_a, segments_a = batch
		if (device != None):
			inputs_a = inputs_a.to(device)
			mask_a = mask_a.to(device)
			segments_a = segments_a.to(device)
		embeddings = bert_encoder.encode(input_ids=inputs_a, token_type_ids=segments_a, attention_mask=mask_a, word_level=False)
		embeddings = embeddings.cpu().detach().numpy()
		all_embeddings.append(embeddings)
	all_embeddings = np.concatenate(all_embeddings, axis=0)
	return all_embeddings


def extract_embeddings_pair(bert_encoder, tokenizer, examples, max_seq_length, device, 
		load, task, label_list, output_mode, norm, word_level=False):
	'''Encode paired examples into BERT embeddings in batches.
	   Used in the computation of gender bias direction.
	   Save computed embeddings under saved_embs/.
	'''
	emb_loc_a = 'saved_embs/num%d_a_%s.pkl' % (len(examples), task)
	emb_loc_b = 'saved_embs/num%d_b_%s.pkl' % (len(examples), task)
	if os.path.isfile(emb_loc_a) and os.path.isfile(emb_loc_b) and load:
		with open(emb_loc_a, 'rb') as f:
			all_embeddings_a = pickle.load(f)
		with open(emb_loc_b, 'rb') as f:
			all_embeddings_b = pickle.load(f)
		print ('preprocessed embeddings loaded from:', emb_loc_a, emb_loc_b)
	else:
		features = convert_examples_to_dualfeatures(
			examples, label_list, max_seq_length, tokenizer, output_mode)
		all_inputs_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
		all_mask_a = torch.tensor([f.mask_a for f in features], dtype=torch.long)
		all_segments_a = torch.tensor([f.segments_a for f in features], dtype=torch.long)
		all_inputs_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
		all_mask_b = torch.tensor([f.mask_b for f in features], dtype=torch.long)
		all_segments_b = torch.tensor([f.segments_b for f in features], dtype=torch.long)

		data = TensorDataset(all_inputs_a, all_inputs_b, all_mask_a, all_mask_b, all_segments_a, all_segments_b)
		dataloader = DataLoader(data, batch_size=32, shuffle=False)
		all_embeddings_a = []
		all_embeddings_b = []
		for step, batch in enumerate(tqdm(dataloader)):
			inputs_a, inputs_b, mask_a, mask_b, segments_a, segments_b = batch
			if (device != None):
				inputs_a = inputs_a.to(device)
				mask_a = mask_a.to(device)
				segments_a = segments_a.to(device)
				inputs_b = inputs_b.to(device)
				mask_b = mask_b.to(device)
				segments_b = segments_b.to(device)
			embeddings_a = bert_encoder.encode(input_ids=inputs_a, token_type_ids=segments_a, attention_mask=mask_a, word_level=False)
			embeddings_b = bert_encoder.encode(input_ids=inputs_b, token_type_ids=segments_b, attention_mask=mask_b, word_level=False)

			embeddings_a /= torch.norm(embeddings_a, dim=-1, keepdim=True)
			embeddings_b /= torch.norm(embeddings_b, dim=-1, keepdim=True)
			if not torch.isnan(embeddings_a).any() and not torch.isnan(embeddings_b).any():
				embeddings_a = embeddings_a.cpu().detach().numpy()
				embeddings_b = embeddings_b.cpu().detach().numpy()
				all_embeddings_a.append(embeddings_a)
				all_embeddings_b.append(embeddings_b)

		all_embeddings_a = np.concatenate(all_embeddings_a, axis=0)
		all_embeddings_b = np.concatenate(all_embeddings_b, axis=0)

		with open(emb_loc_a, 'wb') as f:
			pickle.dump(all_embeddings_a, f)
		with open(emb_loc_b, 'wb') as f:
			pickle.dump(all_embeddings_b, f)

		print ('preprocessed embeddings saved to:', emb_loc_a, emb_loc_b)

	means = (all_embeddings_a + all_embeddings_b) / 2.0
	all_embeddings_a -= means
	all_embeddings_b -= means
	all_embeddings = np.concatenate([all_embeddings_a, all_embeddings_b], axis=0)
	return all_embeddings


def doPCA(matrix, num_components=10):
	pca = PCA(n_components=num_components, svd_solver="auto")
	pca.fit(matrix) # Produce different results each time...
	return pca


def get_def_examples(def_pairs):
	'''Construct definitional examples from definitional pairs.'''
	def_examples = []
	for group_id in def_pairs:
		def_group = def_pairs[group_id]
		f_sents = def_group['f']
		m_sents = def_group['m']
		for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
			def_examples.append(InputExample(guid='{}-{}'.format(group_id, sent_id), 
				text_a=sent_a, text_b=sent_b, label=None))
	return def_examples


def compute_gender_dir(device, tokenizer, bert_encoder, def_pairs, max_seq_length, k, load, task, word_level=False, keepdims=False):
	'''Compute gender bias direction from definitional sentence pairs.'''
	def_examples = get_def_examples(def_pairs) # 1D list where 2i and 2i+1 are a pair

	all_embeddings = extract_embeddings_pair(bert_encoder, tokenizer, def_examples, max_seq_length, device, load, task, 
		label_list=None, output_mode=None, norm=True, word_level=word_level)
	gender_dir = doPCA(all_embeddings).components_[:k]
	if (not keepdims):
		gender_dir = np.mean(gender_dir, axis=0)
	logger.info("gender direction={} {} {}".format(gender_dir.shape,
			type(gender_dir), gender_dir[:10]))
	return gender_dir


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
	"""Loads a data file into a list of input features."""
	'''
	output_mode: classification or regression
	'''	
	if (label_list != None):
		label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0       0 0    1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0    0   0   0   0   0   0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambiguously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert(len(input_ids) == max_seq_length)
		assert(len(input_mask) == max_seq_length)
		assert(len(segment_ids) == max_seq_length)

		if (label_list != None):
			if output_mode == "classification":
				label_id = label_map[example.label]
			elif output_mode == "regression":
				label_id = float(example.label)
			else:
				raise KeyError(output_mode)
		else:
			label_id = None

		features.append(
				InputFeatures(tokens=tokens,
							  input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_id))
	return features


def convert_examples_to_dualfeatures(examples, label_list, max_seq_length, tokenizer, output_mode):
	"""Loads a data file into a list of dual input features."""
	'''
	output_mode: classification or regression
	'''	
	features = []
	for (ex_index, example) in enumerate(tqdm(examples)):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)
		# truncate length
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[:(max_seq_length - 2)]

		tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
		segments_a = [0] * len(tokens_a)
		input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
		mask_a = [1] * len(input_ids_a)
		padding_a = [0] * (max_seq_length - len(input_ids_a))
		input_ids_a += padding_a
		mask_a += padding_a
		segments_a += padding_a
		assert(len(input_ids_a) == max_seq_length)
		assert(len(mask_a) == max_seq_length)
		assert(len(segments_a) == max_seq_length)

		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			if len(tokens_b) > max_seq_length - 2:
				tokens_b = tokens_b[:(max_seq_length - 2)]

			tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
			segments_b = [0] * len(tokens_b)
			input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
			mask_b = [1] * len(input_ids_b)
			padding_b = [0] * (max_seq_length - len(input_ids_b))
			input_ids_b += padding_b
			mask_b += padding_b
			segments_b += padding_b
			assert(len(input_ids_b) == max_seq_length)
			assert(len(mask_b) == max_seq_length)
			assert(len(segments_b) == max_seq_length)
		else:
			input_ids_b = None
			mask_b = None
			segments_b = None

		features.append(
				DualInputFeatures(input_ids_a=input_ids_a,
						     	  input_ids_b=input_ids_b,
								  mask_a=mask_a,
								  mask_b=mask_b,
								  segments_a=segments_a,
								  segments_b=segments_b))
	return features


def simple_accuracy(preds, labels):
	return (preds == labels).mean()


def pearson_and_spearman(preds, labels):
	pearson_corr = pearsonr(preds, labels)[0]
	spearman_corr = spearmanr(preds, labels)[0]
	return {
		"pearson": pearson_corr,
		"spearmanr": spearman_corr,
		"corr": (pearson_corr + spearman_corr) / 2,
	}


def compute_metrics(task_name, preds, labels):
	assert len(preds) == len(labels)
	if task_name == "cola":
		return {"mcc": matthews_corrcoef(labels, preds)}
	elif task_name == "sst-2":
		return {"acc": simple_accuracy(preds, labels)}
	elif task_name == "qnli":
		return {"acc": simple_accuracy(preds, labels)}
	else:
		raise KeyError(task_name)


def parse_args():
	'''Parse command line arguments.'''
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--bert_model", default="bert-base-uncased", type=str, 
						choices = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
						"bert-large-cased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased",
						"bert-base-chinese"],
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
						"bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
						"bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--task_name",
						default=None,
						type=str,
						required=True,
						help="The name of the task to train.")
	parser.add_argument("--output_dir",
						default=None,
						type=str,
						help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--cache_dir",
						default="",
						type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--resume_model_path",
						type=str,
						default="",
						help="Whether to resume from a model.")
	parser.add_argument("--do_train",
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_lower_case",
						action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--normalize",
						action='store_true',
						help="Set this flag if you want embeddings normalized.")
	parser.add_argument("--tune_bert",
						action='store_true',
						help="Set this flag if you want to fine-tune bert model.")
	parser.add_argument("--debias",
						action='store_true',
						help="Set this flag if you want embeddings debiased.")
	parser.add_argument("--no_save",
						action='store_true',
						help="Set this flag if you don't want to save any results.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=2e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

	parser.add_argument("--def_pairs_name", default="all", type=str,
						help="Name of definitional sentence pairs.")
	parser.add_argument("--num_dimension", "-k", type=int, default=1,
						help="dimensionality of bias subspace")
	args = parser.parse_args()

	if (args.output_dir == None):
		args.output_dir = os.path.join('results', args.task_name, args.bert_model)
	print("output_dir={}".format(args.output_dir))

	if (args.do_lower_case and 'uncased' not in args.bert_model):
		raise ValueError("The pre-trained model you are loading is a cased model but you have not set "
						  "`do_lower_case` to False.")
	if (not args.do_lower_case and 'uncased' in args.bert_model):
		raise ValueError("The pre-trained model you are loading is an uncased model but you have not set "
						  "`do_lower_case` to True.")

	return args


def get_tokenizer_encoder(args, device=None):
	'''Return BERT tokenizer and encoder based on args. Used in eval_bias.py.'''
	print("get tokenizer from {}".format(args.model_path))
	tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
	cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
	model_weights_path = args.model_path

	model = BertForSequenceClassification.from_pretrained(model_weights_path,
			  cache_dir=cache_dir,
			  num_labels=2,
			  normalize=args.normalize,
			  tune_bert=args.tune_bert)
	if (device != None): model.to(device)
	bert_encoder = BertEncoder(model, device)

	return tokenizer, bert_encoder


def get_encodings(args, encs, tokenizer, bert_encoder, gender_space, device, 
		word_level=False, specific_set=None):
	'''Extract BERT embeddings from encodings dictionary.
	   Perform the debiasing step if debias is specified in args.
	'''
	if (word_level): assert(specific_set != None)

	logger.info("Get encodings")
	logger.info("Debias={}".format(args.debias))

	examples_dict = dict()
	for key in ['targ1', 'targ2', 'attr1', 'attr2']:
		texts = encs[key]['examples']
		category = encs[key]['category'].lower()
		examples = []
		encs[key]['text_ids'] = dict()
		for i, text in enumerate(texts):
			examples.append(InputExample(guid='{}'.format(i), text_a=text, text_b=None, label=None))
			encs[key]['text_ids'][i] = text
		examples_dict[key] = examples
		all_embeddings = extract_embeddings(bert_encoder, tokenizer, examples, args.max_seq_length, device, 
					label_list=None, output_mode=None, norm=False, word_level=word_level)

		logger.info("Debias category {}".format(category))

		emb_dict = {}
		for index, emb in enumerate(all_embeddings):
			emb /= np.linalg.norm(emb)
			if (args.debias and not category in {'male','female'}): # don't debias gender definitional sentences
				emb = my_we.dropspace(emb, gender_space)
			emb /= np.linalg.norm(emb) # Normalization actually doesn't affect e_size
			emb_dict[index] = emb

		encs[key]['encs'] = emb_dict
	return encs


def prepare_model_and_bias(args, device, num_labels, cache_dir):
	'''Return model and gender direction (computed by resume_model_path)'''
	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	# a. load pretrained model and compute gender direction
	model_weights_path = args.bert_model if (args.resume_model_path == "") else args.resume_model_path
	logger.info("Initialize model with {}".format(model_weights_path))
	model = BertForSequenceClassification.from_pretrained(model_weights_path,
			  cache_dir=cache_dir,
			  num_labels=num_labels,
			  normalize=args.normalize,
			  tune_bert=args.tune_bert).to(device)
	gender_dir = None
	if (args.debias):
		bert_encoder = BertEncoder(model, device)
		def_pairs = get_def_pairs(args.def_pairs_name)
		gender_dir = compute_gender_dir(device, tokenizer, bert_encoder, 
			def_pairs, args.max_seq_length, k=args.num_dimension, load=True, task='pretrained')
		gender_dir = torch.tensor(gender_dir, dtype=torch.float, device=device)

	return model, tokenizer, gender_dir


def prepare_model_and_pretrained_bias(args, device, num_labels, cache_dir):
	'''Return model and gender direction (computed by pretrained bert)'''
	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	# a. load pretrained model and compute gender direction
	model_pretrained = BertForSequenceClassification.from_pretrained(args.bert_model,
			  cache_dir=cache_dir,
			  num_labels=num_labels,
			  normalize=args.normalize,
			  tune_bert=args.tune_bert).to(device)
	gender_dir_pretrained = None
	if (args.debias):
		bert_encoder_pretrained = BertEncoder(model_pretrained, device)
		def_pairs = get_def_pairs(args.def_pairs_name)
		gender_dir_pretrained = compute_gender_dir(device, tokenizer, bert_encoder_pretrained, 
			def_pairs, args.max_seq_length, k=args.num_dimension, load=True, task='pretrained')
		gender_dir_pretrained = torch.tensor(gender_dir_pretrained, dtype=torch.float, device=device)

	if (args.resume_model_path == ""):
		model_weights_path = args.bert_model
	else:
		model_weights_path = args.resume_model_path
		logger.info("Resume training from {}".format(model_weights_path))
	
	# b. Load model for training
	if (args.do_train and args.resume_model_path != ""):
		del model_pretrained
		model = BertForSequenceClassification.from_pretrained(args.resume_model_path,
				  cache_dir=cache_dir,
				  num_labels=num_labels,
				  normalize=args.normalize,
				  tune_bert=args.tune_bert).to(device)
	else:
		model = model_pretrained

	return model, tokenizer, gender_dir_pretrained


def prepare_optimizer(args, model, num_train_optimization_steps):
	'''Initialize and return optimizer.'''
	# Prepare optimizer
	logger.info("Prepare optimizer {} fine-tuning".format("with" if args.tune_bert else "without"))
	if (args.tune_bert):
		param_optimizer = list(model.named_parameters()) # include all parameters
	else:
		param_optimizer = list(model.classifier.named_parameters()) # only the classification head
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	optimizer = BertAdam(optimizer_grouped_parameters,
						 lr=args.learning_rate,
						 warmup=args.warmup_proportion,
						 t_total=num_train_optimization_steps)

	return optimizer


def main():
	'''Fine-tune BERT on the specified task and evaluate on dev set.'''
	args = parse_args()

	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd
		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	processors = {
		"cola": ColaProcessor,
		"sst-2": Sst2Processor,
		"qnli": QnliProcessor
	}

	output_modes = {
		"cola": "classification",
		"sst-2": "classification",
		"qnli": "classification"
	}

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')

	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
							args.gradient_accumulation_steps))

	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	if (not args.no_save):
		if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
			raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)

	task_name = args.task_name.lower()
	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()
	output_mode = output_modes[task_name]

	label_list = processor.get_labels()
	num_labels = len(label_list)

	train_examples = None
	num_train_optimization_steps = None

	cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

	if (args.do_train): 
		# Prepare training examples, model, tokenizer, optimizer, and bias direction.
		train_examples = processor.get_train_examples(args.data_dir)
		num_train_optimization_steps = int(
			len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
		if args.local_rank != -1:
			num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

		model, tokenizer, gender_dir_pretrained = prepare_model_and_bias(args, device, num_labels, cache_dir)	
		optimizer = prepare_optimizer(args, model, num_train_optimization_steps)

	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0
	if args.do_train:
		# start training
		logger.info("Prepare training features")
		train_features = convert_examples_to_features(
			train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_optimization_steps)
		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

		if output_mode == "classification":
			all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
		elif output_mode == "regression":
			all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_data)
		else:
			train_sampler = DistributedSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

		model.classifier.train()
		for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
			epoch_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids = batch

				# define a new function to compute loss values for both output_modes
				logits = model(input_ids, segment_ids, input_mask, remove_bias=args.debias, bias_dir=gender_dir_pretrained)

				if output_mode == "classification":
					loss_fct = CrossEntropyLoss()
					loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
				elif output_mode == "regression":
					loss_fct = MSELoss()
					loss = loss_fct(logits.view(-1), label_ids.view(-1))

				if n_gpu > 1:
					loss = loss.mean() # mean() to average on multi-gpu.
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps

				loss.backward()

				tr_loss += loss.item()
				epoch_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1
				if (step + 1) % args.gradient_accumulation_steps == 0:
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1
			epoch_loss /= len(train_dataloader)
			print("Epoch {}: loss={}".format(epoch, epoch_loss))

		if not args.no_save:
			# Save a trained model, configuration and tokenizer
			model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

			# If we save using the predefined names, we can load using `from_pretrained`
			output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
			output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

			torch.save(model_to_save.state_dict(), output_model_file)
			model_to_save.config.to_json_file(output_config_file)
			tokenizer.save_vocabulary(args.output_dir)

	if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		if (not args.do_train):
			# Load a trained model and vocabulary that you have fine-tuned
			model = BertForSequenceClassification.from_pretrained(args.output_dir,
					  cache_dir=cache_dir,
					  num_labels=num_labels,
					  normalize=args.normalize,
					  tune_bert=args.tune_bert)
			tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
			model.to(device)
		# Get gender direction
		gender_dir_tuned = None
		if args.debias:
			bert_encoder = BertEncoder(model, device)
			def_pairs = get_def_pairs(args.def_pairs_name)
			gender_dir_tuned = compute_gender_dir(device, tokenizer, bert_encoder, def_pairs, args.max_seq_length, k=args.num_dimension, load=False, task=args.task_name)
			gender_dir_tuned = torch.tensor(gender_dir_tuned, dtype=torch.float, device=device)

		eval_examples = processor.get_dev_examples(args.data_dir)
		eval_features = convert_examples_to_features(
			eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(eval_examples))
		logger.info("  Batch size = %d", args.eval_batch_size)
		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

		if output_mode == "classification":
			all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
		elif output_mode == "regression":
			all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

		all_sample_ids = torch.arange(len(eval_features), dtype=torch.long)

		eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sample_ids)
		# Run prediction for full data
		eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)

		model.eval()
		eval_loss = 0
		nb_eval_steps = 0
		preds = []

		for input_ids, input_mask, segment_ids, label_ids, sample_ids in tqdm(eval_dataloader, desc="Evaluating"):
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			label_ids = label_ids.to(device)

			with torch.no_grad():
				logits = model(input_ids, segment_ids, input_mask, 
					labels=None, remove_bias=args.debias, bias_dir=gender_dir_tuned)

			# create eval loss and other metric required by the task
			if output_mode == "classification":
				loss_fct = CrossEntropyLoss()
				tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
			elif output_mode == "regression":
				loss_fct = MSELoss()
				tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
			
			eval_loss += tmp_eval_loss.mean().item()
			nb_eval_steps += 1
			if len(preds) == 0:
				preds.append(logits.detach().cpu().numpy())
			else:
				preds[0] = np.append(
					preds[0], logits.detach().cpu().numpy(), axis=0)

		eval_loss = eval_loss / nb_eval_steps
		preds = preds[0]
		if output_mode == "classification":
			preds = np.argmax(preds, axis=1)
		elif output_mode == "regression":
			preds = np.squeeze(preds)
		result = compute_metrics(task_name, preds, all_label_ids.numpy())
		loss = tr_loss/global_step if (args.do_train and global_step > 0) else None

		result['eval_loss'] = eval_loss
		result['global_step'] = global_step
		result['loss'] = loss

		output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
		with open(output_eval_file, "w") as writer:
			logger.info("***** Eval results *****")
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				if (not args.no_save):
					writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
	main()






