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

import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.decomposition import PCA

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertModel, BertConfig
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
# from bias_data.def_sent_pairs import full_def_sent_pairs, thisis_def_sent_pairs, expanded_thisis
# from bias_data.more_def_sent_pairs import full_def_sent_pairs
from bias_data.def_sent_pairs import pairs_dict

from allennlp.commands.elmo import ElmoEmbedder

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config(dict):
		def __init__(self, **kwargs):
				super().__init__(**kwargs)
				for k, v in kwargs.items():
						setattr(self, k, v)
		
		def set(self, key, val):
				self[key] = val
				setattr(self, key, val)
				
config = Config(
		testing=True,
		seed=1,
		batch_size=64,
		lr=3e-4,
		epochs=2,
		hidden_sz=64,
		max_seq_len=100, # necessary to limit memory usage
		max_vocab_size=100000,
)

OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file=OPTIONS_FILE, weight_file=WEIGHT_FILE,
	do_layer_norm=False, dropout=0.0, num_output_representations=1).to(device)

def tokenizer(x: str):
		return [w.text for w in
						SpacyWordSplitter(language='en_core_web_sm', 
															pos_tags=False).split_words(x)[:config.max_seq_len]]

class LSTMClassifier(nn.Module):
	def __init__(self, elmo, num_labels, device, normalize=False):
		super(LSTMClassifier, self).__init__()
		self.elmo = elmo.to(device)
		self.input_dim = 1024
		self.hidden_size = 512
		self.num_layers = 1
		self.num_labels = num_labels
		self.dropout = nn.Dropout(0.1)

		self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, 
			num_layers=self.num_layers , batch_first=True)
		self.classifier = nn.Linear(self.hidden_size * self.num_layers, num_labels)

		# self.hidden2lable = nn.Linear(self.input_dim, num_labels)

		self.act = nn.Tanh()
		self.device = device
		self.normalize = normalize
		logger.info("Normalize={}".format(normalize))

	def drop_bias(self, u, v):
		return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)

	def forward(self, embeddings, remove_bias=False, bias_dir=None):
		# embeddings: batch_size x T x embed_size
		batch_size = embeddings.shape[0]
		T = embeddings.shape[1]
		if (self.normalize):
			embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
		if (remove_bias):
			embeddings = embeddings.view(-1, self.input_dim)
			embeddings = self.drop_bias(embeddings, bias_dir)
			embeddings = embeddings.view(batch_size, T, self.input_dim)
			embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
		# Gradient starts from here
		embeddings = self.dropout(embeddings)
		_, (h, c) = self.rnn(embeddings)
		# h: 2xBxH
		h = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
		logits = self.classifier(h)

		return logits

class FullyConnectedClassifier(nn.Module):
	def __init__(self, num_labels, hidden_dims, device, normalize=False):
		super(FullyConnectedClassifier, self).__init__()
		self.input_dim = 1024
		self.num_labels = num_labels
		self.dropout = nn.Dropout(0.1)
		layers = []
		hidden_dims.append(num_labels)
		prev_hidden_dim = self.input_dim
		for i in range(len(hidden_dims)):
			hidden_dim = hidden_dims[i]
			layers.append(nn.Linear(prev_hidden_dim, hidden_dim))
			prev_hidden_dim = hidden_dim
		self.layers = nn.ModuleList(layers)

		self.act = nn.Tanh()
		self.device = device
		self.normalize = normalize
		logger.info("Normalize={}".format(normalize))

	def drop_bias(self, u, v):
		return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)

	def forward(self, embeddings, remove_bias=False, bias_dir=None):
		# embeddings: batch_size x embed_size
		# Detach from here
		if (self.normalize):
			embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
		embeddings -= torch.mean(embeddings, dim=-1, keepdim=True)
		embeddings /= torch.std(embeddings, dim=-1, keepdim=True)
		if (remove_bias):
			embeddings = self.drop_bias(embeddings, bias_dir)
			embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
		# Gradient starts from here
		embeddings = self.dropout(embeddings)
		for layer in self.layers:
			embeddings = self.act(embeddings)
			embeddings = layer(embeddings)
		logits = embeddings

		return logits

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

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
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


class ColaElmoProcessor(DataProcessor):
	"""Processor for the CoLA data set (GLUE version)."""

	def get_train_examples(self, data_dir):
		data_file = os.path.join(data_dir, "{};{}.npy".format("cola", "train"))
		examples = np.load(data_file)
		return examples

	def get_dev_examples(self, data_dir):
		data_file = os.path.join(data_dir, "{};{}.npy".format("cola", "dev"))
		examples = np.load(data_file)
		return examples

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

class Sst2ElmoProcessor(DataProcessor):
	"""Processor for the SST-2 data set (GLUE version)."""

	def get_train_examples(self, data_dir):
		data_file = os.path.join(data_dir, "{};{}.npy".format("sst-2", "train"))
		examples = np.load(data_file)
		return examples

	def get_dev_examples(self, data_dir):
		data_file = os.path.join(data_dir, "{};{}.npy".format("sst-2", "dev"))
		examples = np.load(data_file)
		return examples

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

class ElmoDataset(Dataset):
	def __init__(self, examples):
		self.examples = examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		example = self.examples[idx]
		sent =  torch.tensor(example[0], dtype=torch.float)
		label = torch.tensor(example[1], dtype=torch.long)
		return sent, label

class SentenceDataset(Dataset):
	def __init__(self, examples):
		self.examples = examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		example = self.examples[idx]
		sent = example.text_a
		label = example.label
		return sent, label

def my_collate(batch):
	sentences = [tokenizer(item[0]) for item in batch]
	character_ids = batch_to_ids(sentences).to(device)
	elmo_dir = elmo(character_ids)
	embeddings = elmo_dir['elmo_representations'][0]

	labels = [item[1] for item in batch]
	labels = torch.LongTensor(labels, device=device)
	return embeddings, labels

def get_def_examples(def_pairs_name):
	def_pairs = pairs_dict[def_pairs_name]
	def_examples = []
	for i, pair in enumerate(def_pairs):
		sentA = pair[0]
		sentB = pair[1]
		def_examples.append(InputExample(guid='{}-a'.format(i), 
			text_a=sentA, text_b=None, label=None))
		def_examples.append(InputExample(guid='{}-b'.format(i), 
			text_a=sentB, text_b=None, label=None))

	# for i in range(10):
	# 	example = def_examples[i]
	# 	print(example.guid, example.text_a)
	return def_examples

def doPCA(pairs, num_components = 10):
	matrix = []
	for a, b in pairs:
		center = (a + b)/2
		matrix.append(a - center)
		matrix.append(b - center)
	matrix = np.array(matrix)
	pca = PCA(n_components=num_components, svd_solver="full")
	pca.fit(matrix) # Produce different results each time...
	return pca


def simple_accuracy(preds, labels):
	return (preds == labels).mean()


def acc_and_f1(preds, labels):
	acc = simple_accuracy(preds, labels)
	f1 = f1_score(y_true=labels, y_pred=preds)
	return {
		"acc": acc,
		"f1": f1,
		"acc_and_f1": (acc + f1) / 2,
	}


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
	elif task_name == "mrpc":
		return acc_and_f1(preds, labels)
	elif task_name == "sts-b":
		return pearson_and_spearman(preds, labels)
	elif task_name == "qqp":
		return acc_and_f1(preds, labels)
	elif task_name == "mnli":
		return {"acc": simple_accuracy(preds, labels)}
	elif task_name == "mnli-mm":
		return {"acc": simple_accuracy(preds, labels)}
	elif task_name == "qnli":
		return {"acc": simple_accuracy(preds, labels)}
	elif task_name == "rte":
		return {"acc": simple_accuracy(preds, labels)}
	elif task_name == "wnli":
		return {"acc": simple_accuracy(preds, labels)}
	else:
		raise KeyError(task_name)

def parse_args():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--bert_model", default=None, type=str, required=True,
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
						help="Set this flag if you want embeddings normalized.")
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
						default=0.1,
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
						default=102,
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

	parser.add_argument("--def_pairs_name", default="large_real", type=str,
						help="Name of definitional sentence pairs.")
	parser.add_argument("--weights_name", default="model_weights", type=str)

	parser.add_argument("--overwrite",
						action='store_true',
						help="Overwrite output directory if it already exists")
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

def correct_count(A, B):
	count = np.sum(A == B)
	return count

def load_gender_dir():
	filename = os.path.join("elmo_data", "gender_dir.npy")
	gender_dir = np.load(filename)
	return gender_dir

def main():
	args = parse_args()

	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd
		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	processors = {
		"cola": ColaProcessor,
		"sst-2": Sst2Processor
	}

	output_modes = {
		"cola": "classification",
		"sst-2": "classification"
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

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite and not args.no_save:
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

	if args.do_train:
		print("Prepare training examples...")
		train_examples = processor.get_train_examples(args.data_dir)
		num_train_optimization_steps = int(
			len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
		if args.local_rank != -1:
			num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

		# train_dataset = SentenceDataset(train_examples)
		# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
		# 	collate_fn=my_collate)
		# logger.info("Number of batches={}".format(len(train_loader)))
		# model = LSTMClassifier(elmo, len(label_list), device, normalize=True).to(device)
		# for data, label in train_loader:
		# 	data = data.to(device)
		# 	# logger.info("data={} label={}".format(len(data), label))
		# 	logits = model(data)
		# 	logger.info("logits={}".format(logits.shape))
		# 	return


	hidden_dims = [512, 256, 128, 64, 16]
	model = LSTMClassifier(elmo, len(label_list), device, normalize=True).to(device)
	if args.fp16:
		model.half()
	model.to(device)
	if args.local_rank != -1:
		try:
			from apex.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		model = DDP(model)
	elif n_gpu > 1:
		model = nn.DataParallel(model)

	gender_dir = None
	if args.debias:
		gender_dir = load_gender_dir()
		gender_dir = torch.tensor(gender_dir, dtype=torch.float, device=device)
		logger.info("Gender direction: {}".format(gender_dir[:10]))

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

	loss_fct = CrossEntropyLoss() if (output_mode == "classification") else MSELoss()

	global_step = 0
	tr_loss = 0
	if args.do_train:

		train_dataset = SentenceDataset(train_examples)
		train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
			collate_fn=my_collate)
		eval_examples = processor.get_dev_examples(args.data_dir)
		eval_dataset = SentenceDataset(eval_examples)
		eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
							collate_fn=my_collate)

		model.train()
		best_epoch_loss = 0.
		best_metric = 0.
		best_result = dict()
		for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
			epoch_loss = 0
			all_correct = 0
			model.train()
			for step, (data, label_ids) in enumerate(tqdm(train_dataloader, desc="Iteration")):
				label_ids = label_ids.to(device)
				data = data.to(device)

				logits = model(data, remove_bias=args.debias, bias_dir=gender_dir)

				# logger.info("logits={}".format(logits))
				predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)
				all_correct += correct_count(predictions, label_ids.cpu().numpy())

				if output_mode == "classification":
					loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
				elif output_mode == "regression":
					loss = loss_fct(logits.view(-1), label_ids.view(-1))

				loss.backward()

				tr_loss += loss.item()
				epoch_loss += loss.item()

				if (step + 1) % args.gradient_accumulation_steps == 0:
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1
			epoch_loss /= len(train_dataloader)
			# Evaluation
			model.eval()
			eval_loss = 0
			nb_eval_steps = 0
			preds = []
			all_label_ids = []

			for data, label_ids in tqdm(eval_dataloader, desc="Evaluating"):

				data = data.to(device)
				label_ids = label_ids.to(device)
				all_label_ids.append(label_ids.cpu().numpy())

				with torch.no_grad():
					logits = model(data, remove_bias=args.debias, bias_dir=gender_dir)

				# logger.info("logits={}".format(logits))
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
			# logger.info("preds={}".format(preds[:10]))
			preds = preds[0]
			if output_mode == "classification":
				preds = np.argmax(preds, axis=1)
			elif output_mode == "regression":
				preds = np.squeeze(preds)

			all_label_ids = np.concatenate(all_label_ids)
			result = compute_metrics(task_name, preds, all_label_ids)
			metric = result[list(result.keys())[0]]
			loss = tr_loss/global_step if args.do_train else None

			result['eval_loss'] = eval_loss
			result['global_step'] = global_step
			result['loss'] = loss
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))

			if (metric > best_metric):
				best_metric = metric
				best_result = result

			if (output_mode == "classification"):
				train_acc = all_correct / len(train_examples)
				print("Epoch {}: loss={} acc={}".format(epoch, epoch_loss, train_acc))
			elif (output_mode == "regression"):
				print("Epoch {}: loss={}".format(epoch, epoch_loss))
			if (epoch_loss < best_epoch_loss):
				best_epoch_loss = epoch_loss
			else:
				scheduler.step()

	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Save a trained model, configuration and tokenizer

		# If we save using the predefined names, we can load using `from_pretrained`
		output_model_file = os.path.join(args.output_dir, args.weights_name)
		if (not args.no_save):
			torch.save(model.state_dict(), output_model_file)

	# Load a trained model that you have fine-tuned
	if (args.do_eval and not args.do_train):
		states = torch.load(args.resume_model_path)
		model.load_state_dict(states["model"])

	if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Get gender direction
		eval_examples = processor.get_dev_examples(args.data_dir)
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(eval_examples))
		logger.info("  Batch size = %d", args.eval_batch_size)

		eval_dataset = SentenceDataset(eval_examples)
		eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
							collate_fn=my_collate)

		model.eval()
		eval_loss = 0
		nb_eval_steps = 0
		preds = []

		all_label_ids = []

		for data, label_ids in tqdm(eval_dataloader, desc="Evaluating"):

			data = data.to(device)
			label_ids = label_ids.to(device)
			all_label_ids.append(label_ids.cpu().numpy())

			with torch.no_grad():
				logits = model(data, remove_bias=args.debias, bias_dir=gender_dir)

			# logger.info("logits={}".format(logits))
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
		# logger.info("preds={}".format(preds[:10]))
		preds = preds[0]
		if output_mode == "classification":
			preds = np.argmax(preds, axis=1)
		elif output_mode == "regression":
			preds = np.squeeze(preds)

		all_label_ids = np.concatenate(all_label_ids)
		logger.info("preds={} {}".format(len(preds), preds[:20]))
		logger.info("label={} {}".format(len(all_label_ids), all_label_ids[:20]))
		result = compute_metrics(task_name, preds, all_label_ids)
		loss = tr_loss/global_step if args.do_train else None

		result['eval_loss'] = eval_loss
		result['global_step'] = global_step
		result['loss'] = loss

		output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
		if (not args.no_save): writer = open(output_eval_file, "w")
		result = best_result
		logger.info("***** Eval results *****")
		for key in sorted(result.keys()):
			logger.info("  %s = %s", key, str(result[key]))
			if (not args.no_save):
				writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
	main()

'''
cola biased:
CUDA_VISIBLE_DEVICES=2 python run_elmo.py  --output_dir elmo-results/CoLA-lstm --task_name CoLA  --do_eval   --do_lower_case   --data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/CoLA   --bert_model bert-base-uncased   --max_seq_length 128   --train_batch_size 32   --learning_rate 0.001   --num_train_epochs 50.0   --normalize --do_train
CUDA_VISIBLE_DEVICES=2 \
python run_elmo.py --output_dir elmo-results/CoLA-lstm-biased \
--task_name CoLA  \
--do_eval  \
--do_lower_case \
--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/CoLA \
--bert_model bert-base-uncased \
--max_seq_length 128  \
--train_batch_size 32 \
--learning_rate 0.001   \
--num_train_epochs 50.0   \
--normalize \
--do_train

mcc: 39.1

cola debias:
CUDA_VISIBLE_DEVICES=3 \
python run_elmo.py --output_dir elmo-results/CoLA-lstm-debiased \
--debias \
--task_name CoLA  \
--do_eval  \
--do_train \
--do_lower_case \
--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/CoLA \
--bert_model bert-base-uncased \
--max_seq_length 128  \
--train_batch_size 32 \
--learning_rate 0.001   \
--num_train_epochs 7.0   \
--normalize \
--debias

sst biased:
screen: elmo-sst-biased
CUDA_VISIBLE_DEVICES=0 \
python run_elmo.py --output_dir elmo-results/SST-2-lstm-biased \
--task_name SST-2  \
--do_eval  \
--do_lower_case \
--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/SST-2 \
--bert_model bert-base-uncased \
--max_seq_length 128  \
--train_batch_size 32 \
--learning_rate 0.001   \
--num_train_epochs 50.0   \
--normalize \
--do_train

sst debiased:
screen: elmo-sst-debias

CUDA_VISIBLE_DEVICES=1 \
python run_elmo.py --output_dir elmo-results/SST-2-lstm-debias \
--task_name SST-2  \
--debias \
--do_eval  \
--do_lower_case \
--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/SST-2 \
--bert_model bert-base-uncased \
--max_seq_length 128  \
--train_batch_size 32 \
--learning_rate 0.001   \
--num_train_epochs 50.0   \
--normalize \
--do_train



'''
