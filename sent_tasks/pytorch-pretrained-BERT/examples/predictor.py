from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.data import Instance

from tqdm import tqdm
from scipy.special import expit # the sigmoid function
import numpy as np
from collections.abc import Iterable

 
def tonp(tsr): return tsr.detach().cpu().numpy()
 
class Predictor:
	def __init__(self, model: Model, iterator: DataIterator,
				 cuda_device: int=-1) -> None:
		self.model = model
		self.iterator = iterator
		self.cuda_device = cuda_device
		 
	def _extract_data(self, batch) -> np.ndarray:
		out_dict = self.model(**batch)
		return expit(tonp(out_dict["class_logits"]))
	 
	def predict(self, ds: Iterable[Instance]) -> np.ndarray:
		pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
		self.model.eval()
		pred_generator_tqdm = tqdm(pred_generator,
								   total=self.iterator.get_num_batches(ds))
		preds = []
		with torch.no_grad():
			for batch in pred_generator_tqdm:
				batch = nn_util.move_to_device(batch, self.cuda_device)
				preds.append(self._extract_data(batch))
		return np.concatenate(preds, axis=0)

from allennlp.data.iterators import BasicIterator
# iterate over the dataset without changing its order
seq_iterator = BasicIterator(batch_size=64)
seq_iterator.index_with(vocab)
 
predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
train_preds = predictor.predict(train_ds) 
test_preds = predictor.predict(test_ds) 

