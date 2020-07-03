from __future__ import print_function, division
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file=options_file, weight_file=weight_file,
	do_layer_norm=False, dropout=0.0, num_output_representations=1)
elmo = elmo.to(device)

# tokens = {""}
# elmo_tokens = tokens.pop("elmo", None)
# elmo_representations = elmo(elmo_tokens)["elmo_representations"]

start_time = time.time()
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences).to(device)
print("character_ids", character_ids.shape, type(character_ids))

embeddings = elmo(character_ids)
elapsed_time = time.time() - start_time
print("time={}".format(elapsed_time))

print(type(embeddings), embeddings.keys())
elmo_representations = embeddings['elmo_representations']
print(len(elmo_representations))
for i in range(len(elmo_representations)):
	e = elmo_representations[i]
	print(type(e), e.shape)

