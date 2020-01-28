import sys
import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

#import encoders.bert as bert
#from data import load_encodings
#OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging as log
import numpy
log.basicConfig(level=log.INFO)

# Load pre-trained model tokenizer (vocabulary)

# model_name = ModelName.BERT.value

# from data: load_encodings
def load_encodings(enc_file):
	encs = dict()
	with h5py.File(enc_file, 'r') as enc_fh:
		for split_name, split in enc_fh.items():
			split_d, split_exs = {}, {}
			for ex, enc in split.items():
				if ex == CATEGORY:
					split_d[ex] = enc.value
				else:
					split_exs[ex] = enc[:]
			split_d["encs"] = split_exs
			encs[split_name] = split_d
	return encs

#bert.py
def load_model(version='bert-base-uncased'):
	tokenizer = BertTokenizer.from_pretrained(version)
	model = BertModel.from_pretrained(version)
	model.eval()
	return model, tokenizer

def encode(model, tokenizer, texts):
	encs = {}
	for text in texts:
		tokenized = tokenizer.tokenize(text)
		indexed = tokenizer.convert_tokens_to_ids(tokenized)
		segment_idxs = [0] * len(tokenized)
		tokens_tensor = torch.tensor([indexed])
		segments_tensor = torch.tensor([segment_idxs])
		enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

		enc = enc[:, 0, :]  # extract the last rep of the first input
		encs[text] = enc.detach().view(-1).numpy()
	return encs


def main(arg_file):
	# model_options = 'version' + args.bert_version

	#model, tokenizer = load_model()
	#encs = load_encodings(enc_file)
	#encs_targ1 = encode(model, tokenizer, encs["targ1"]["examples"])
	#encs_targ2 = encode(model, tokenizer, encs["targ2"]["examples"])
	#encs_attr1 = encode(model, tokenizer, encs["attr1"]["examples"])
	#encs_attr2 = encode(model, tokenizer, encs["attr2"]["examples"])


	#enc = [e for e in encs["targ1"]['encs'].values()][0]
	#d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)
	model, tokenizer = load_model()
	texts = 
	res = encode(model, tokenizer, texts)


if __name__ == '__main__':
	main(def_sent.jsonl)
