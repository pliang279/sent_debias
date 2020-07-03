from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

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

from allennlp.common.checks import ConfigurationError

USE_GPU = torch.cuda.is_available()
DATA_ROOT = Path("../data") / "jigsaw"
torch.manual_seed(config.seed)

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader

label_cols = ["toxic", "severe_toxic", "obscene",
							"threat", "insult", "identity_hate"]

from allennlp.data.fields import TextField, MetadataField, ArrayField

class JigsawDatasetReader(DatasetReader):
		def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
								 token_indexers: Dict[str, TokenIndexer] = None,
								 max_seq_len: Optional[int]=config.max_seq_len) -> None:
				super().__init__(lazy=False)
				self.tokenizer = tokenizer
				self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
				self.max_seq_len = max_seq_len

		@overrides
		def text_to_instance(self, tokens: List[Token]) -> Instance:
				sentence_field = TextField(tokens, self.token_indexers)
				fields = {"tokens": sentence_field}

				return Instance(fields)
		
		@overrides
		def _read(self) -> Iterator[Instance]:
				for sent in sentences:
						yield self.text_to_instance(
								[Token(x) for x in self.tokenizer(sent)]
						)


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer

# the token indexer is responsible for mapping tokens to integers
token_indexer = ELMoTokenCharactersIndexer()


def tokenizer(x: str):
		return [w.text for w in
						SpacyWordSplitter(language='en_core_web_sm', 
															pos_tags=False).split_words(x)[:config.max_seq_len]]

sent = "Hello world!"
print("sent", sent)
print(tokenizer(sent))
tokens = [Token(x) for x in tokenizer(sent)]
for token in tokens: print(type(token))

reader = JigsawDatasetReader(
		tokenizer=tokenizer,
		token_indexers={"tokens": token_indexer}
)

# train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train.csv", "test_proced.csv"])
# val_ds = None

vocab = Vocabulary()

from allennlp.data.iterators import BucketIterator

iterator = BucketIterator(batch_size=config.batch_size,
													sorting_keys=[("tokens", "num_tokens")],
												 )

iterator.index_with(vocab)
batch = next(iter(iterator()))

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder

class BaselineModel(Model):
		def __init__(self, word_embeddings: TextFieldEmbedder,
								 encoder: Seq2VecEncoder,
								 out_sz: int=len(label_cols)):
				super().__init__(vocab)
				self.word_embeddings = word_embeddings
				self.encoder = encoder
				self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
				self.loss = nn.BCEWithLogitsLoss()
				
		def forward(self, tokens: Dict[str, torch.Tensor],
								id: Any, label: torch.Tensor) -> torch.Tensor:
				mask = get_text_field_mask(tokens)
				embeddings = self.word_embeddings(tokens)
				state = self.encoder(embeddings, mask)
				class_logits = self.projection(state)
				
				output = {"class_logits": class_logits}

				return output

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})


from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(), config.hidden_sz, bidirectional=True, batch_first=True))

model = BaselineModel(
    word_embeddings, 
    encoder, 
)

if USE_GPU: model.cuda()
else: model


batch = nn_util.move_to_device(batch, 0 if USE_GPU else -1)
tokens = batch["tokens"]
labels = batch
mask = get_text_field_mask(tokens)

embeddings = model.word_embeddings(tokens)
state = model.encoder(embeddings, mask)
class_logits = model.projection(state)

model(**batch)
loss = model(**batch)["loss"]
loss.backward()
