from __future__ import absolute_import, division, print_function
import numpy as np
import json
import os
import logging
from run_classifier import get_encodings
import weat
import argparse
from scipy import spatial
from bias_data.thisis_tests import *

logger = logging.getLogger(__name__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)


DATA_DIR = "../bias_data/thisis_tests/"

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--resume_model_path",
						type=str,
						default="",
						help="Whether to resume from a model.")
	parser.add_argument("--bert_model", default="bert-base-uncased", type=str, 
						choices = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
						"bert-large-cased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased",
						"bert-base-chinese"],
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
						"bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
						"bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--debias",
						action='store_true',
						help="Whether to debias.")
	parser.add_argument("--equalize",
						action='store_true',
						help="Whether to equalize.")
	args = parser.parse_args()
	args.do_lower_case = True
	args.cache_dir = None
	args.local_rank = -1
	args.max_seq_length = 128
	args.eval_batch_size = 8
	args.n_samples = 100000
	args.parametric = True
	args.tune_bert = False
	args.normalize = True
	return args

def binary_weat(targets, attributes):
	targetOne = []
	targetTwo = []
	for x in targets[0]:
		targetOne.append(_binary_s(x, attributes))
	for y in targets[1]:
		targetTwo.append(_binary_s(y, attributes))

	weat_score = np.absolute(sum(targetOne) - sum(targetTwo))

	wtmp = [_binary_s(t, attributes) for t in targets[0] + targets[1]]
	effect_std = np.std(wtmp)
	num = np.absolute((sum(targetOne)/float(len(targetOne)) - sum(targetTwo)/float(len(targetTwo))))
	effect_size = (num/effect_std)
	return weat_score, effect_size

def _binary_s(target, attributes):
	groupOne = []
	groupTwo = []
	for ai in attributes[0]:
		groupOne.append(spatial.distance.cosine(target, ai))
	for aj in attributes[1]:
		groupTwo.append(spatial.distance.cosine(target, aj))
	return np.absolute(sum(groupOne)/float(len(groupOne)) - sum(groupTwo)/float(len(groupTwo)))

def main():
	args = parse_args()

	results = []
	for test_name in sent_tests.keys():
		encs = sent_tests[test_name]
		encs = get_encodings(args, encs, debias=args.debias, equalize=args.equalize)
		esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
		targ1 = list(encs['targ1']['encs'].values())
		targ2 = list(encs['targ2']['encs'].values())
		attr1 = list(encs['attr1']['encs'].values())
		attr2 = list(encs['attr2']['encs'].values())
		targets = [targ1, targ2]
		attributes = [attr1, attr2]
		weat_score, effect_size = binary_weat(targets, attributes)
		results.append("{}: esize={} pval={} | w_score={} esize={}".format(test_name, 
			esize, pval, weat_score, effect_size))

	for result in results:
		logger.info(result)

'''
full solver
B0: 6, 7, 8, 8b
B0 original:
05/16/2019 00:38:23 - INFO - __main__ -   sent6: esize=-0.3663679874351231 pval=0.7669314946342223 | w_score=0.0076627305575779625 esize=0.1621401765789137
05/16/2019 00:38:23 - INFO - __main__ -   sent6b: esize=-0.5840541175057232 pval=0.8782972617238722 | w_score=0.004246637225151062 esize=0.11605606158408992
05/16/2019 00:38:23 - INFO - __main__ -   sent7: esize=0.4209404871563804 pval=0.19853120819804088 | w_score=0.003362527915409641 esize=0.10991860520874203
05/16/2019 00:38:23 - INFO - __main__ -   sent7b: esize=0.4129914382781003 pval=0.20314903091188435 | w_score=0.06788833439350128 esize=0.7926176504710125
05/16/2019 00:38:23 - INFO - __main__ -   sent8: esize=0.3264266756878826 pval=0.25719976179400533 | w_score=0.015494082655225491 esize=0.5052400023528992
05/16/2019 00:38:23 - INFO - __main__ -   sent8b: esize=0.22805902898718022 pval=0.3255693737726134 | w_score=0.023854218423366547 esize=0.28556825814335596

B0 debias w/o equalize:
05/16/2019 00:41:51 - INFO - __main__ -   sent6: esize=0.06307553684107056 pval=0.4489138955260339 | w_score=0.00250166654586792 esize=0.20251449900420515
05/16/2019 00:41:51 - INFO - __main__ -   sent6b: esize=1.297917715604691 pval=0.0048387687637205645 | w_score=0.009474672377109528 esize=0.8562666395695118
05/16/2019 00:41:51 - INFO - __main__ -   sent7: esize=0.2490305820368855 pval=0.3089662711544813 | w_score=0.0002305933407374828 esize=0.02637406478167739
05/16/2019 00:41:51 - INFO - __main__ -   sent7b: esize=1.1129026209346766 pval=0.013020232673239842 | w_score=0.016755394637584686 esize=1.1494011778701794
05/16/2019 00:41:51 - INFO - __main__ -   sent8: esize=0.058622496583045886 pval=0.4553013530839208 | w_score=0.0025282927921840056 esize=0.27886712919873485
05/16/2019 00:41:51 - INFO - __main__ -   sent8b: esize=-0.06263307526329132 pval=0.5510030959774657 | w_score=0.0008569210767745972 esize=0.06468348379275511

B0 debias w/ equalize:
05/16/2019 00:16:31 - INFO - __main__ -   sent6: esize=-0.42809827818740775 pval=0.8042672889964835 | w_score=1.7029898505271923e-08 esize=0.16724840198235635
05/16/2019 00:16:31 - INFO - __main__ -   sent7: esize=0.7684971609716837 pval=0.06187406251188856 | w_score=3.4059797010543846e-08 esize=0.48507125002616147
05/16/2019 00:16:31 - INFO - __main__ -   sent8: esize=-0.3138594779894112 pval=0.735721337047347 | w_score=1.734723475976807e-18 esize=1.8675606345656648e-11

B1: 6, 6b, 7, 7b, 8, 8b
B1 original:
05/16/2019 00:46:30 - INFO - __main__ -   sent6: esize=0.9650236172136684 pval=0.02633349305739743 | w_score=0.07339226002139712 esize=0.9369842039110486
05/16/2019 00:46:30 - INFO - __main__ -   sent6b: esize=-0.3420411651916362 pval=0.7527247910483025 | w_score=0.1409413879737258 esize=0.35325859985447394
05/16/2019 00:46:30 - INFO - __main__ -   sent7: esize=1.1360113504470482 pval=0.01157738791534261 | w_score=0.055603591525660995 esize=0.5977704790921603
05/16/2019 00:46:30 - INFO - __main__ -   sent7b: esize=-0.5990735345305295 pval=0.8843772166120045 | w_score=0.08512902577058412 esize=0.8312040036609293
05/16/2019 00:46:30 - INFO - __main__ -   sent8: esize=-0.29620045234149806 pval=0.7236064094216347 | w_score=0.044719377998262616 esize=0.6546468254378874
05/16/2019 00:46:30 - INFO - __main__ -   sent8b: esize=-0.1682838245114684 pval=0.6296207824375807 | w_score=0.008814248722046614 esize=0.09289418121679056

B1 debias (perfect!!!):
05/16/2019 00:47:49 - INFO - __main__ -   sent6: esize=-0.5044222070374137 pval=0.8437344926903148 | w_score=0.009628879172461344 esize=0.17207004722215222
05/16/2019 00:47:49 - INFO - __main__ -   sent6b: esize=-0.32692252745925965 pval=0.7438323106544587 | w_score=0.006485331803560257 esize=0.3376502362824175
05/16/2019 00:47:49 - INFO - __main__ -   sent7: esize=0.31167312067463676 pval=0.26563253723998825 | w_score=0.048409838761602264 esize=0.5545753589974809
05/16/2019 00:47:49 - INFO - __main__ -   sent7b: esize=0.3959393897334448 pval=0.21224367456699744 | w_score=0.020783626474440098 esize=0.4740774879408489
05/16/2019 00:47:49 - INFO - __main__ -   sent8: esize=0.060784996235008305 pval=0.4523482917556103 | w_score=0.010748612029211943 esize=0.13067985258054807
05/16/2019 00:47:49 - INFO - __main__ -   sent8b: esize=0.01657790236734332 pval=0.4872167675291519 | w_score=0.008066888898611069 esize=0.20611044264727305

B1 debias more definitional pairs
05/16/2019 01:07:58 - INFO - __main__ -   sent6: esize=0.039758277499222894 pval=0.4700233143432946 | w_score=0.05570252878325321 esize=0.4287683243798206
05/16/2019 01:07:58 - INFO - __main__ -   sent6b: esize=-0.35467680226239806 pval=0.7629081755890068 | w_score=0.1059819720685482 esize=0.40872091233737573
05/16/2019 01:07:58 - INFO - __main__ -   sent7: esize=0.5708663224864025 pval=0.12674583717919163 | w_score=0.06687444448471067 esize=0.4738804435712804
05/16/2019 01:07:58 - INFO - __main__ -   sent7b: esize=0.3054802512493809 pval=0.26958002804721903 | w_score=0.01701696217060089 esize=0.11536044153330566
05/16/2019 01:07:58 - INFO - __main__ -   sent8: esize=-0.2500258185415806 pval=0.6924415031569235 | w_score=0.05449742078781128 esize=0.4146272697764722
05/16/2019 01:07:58 - INFO - __main__ -   sent8b: esize=-0.7480046966245238 pval=0.9331460840425321 | w_score=0.10747185349464417 esize=0.7633052956610736

B1 more fine-grained def pairs
05/16/2019 01:46:44 - INFO - __main__ -   sent6: esize=-0.6325829859113047 pval=0.8969864331897416 | w_score=0.009928315877914512 esize=0.18012675776452072
05/16/2019 01:46:44 - INFO - __main__ -   sent6b: esize=-0.5820675508187966 pval=0.8770935829676374 | w_score=0.20943618938326836 esize=0.6011563283646613
05/16/2019 01:46:44 - INFO - __main__ -   sent7: esize=-0.4875963892501444 pval=0.8340387289000206 | w_score=0.02707205393484649 esize=0.4516433046129664
05/16/2019 01:46:44 - INFO - __main__ -   sent7b: esize=-0.730310886817792 pval=0.9277408310117236 | w_score=0.1542756436392665 esize=1.091921601585275
05/16/2019 01:46:44 - INFO - __main__ -   sent8: esize=-0.03855955501175208 pval=0.5295923962644992 | w_score=0.015318907797336606 esize=0.2524185847208332
05/16/2019 01:46:44 - INFO - __main__ -   sent8b: esize=-0.21768104309881497 pval=0.6678975305424362 | w_score=0.017406773753464222 esize=0.13505063326976682

B2: 6, 7b, 8b
B2 original:
05/16/2019 00:51:22 - INFO - __main__ -   sent6: esize=1.0147410564137351 pval=0.021417929318562554 | w_score=0.02520452652658739 esize=0.4054209518240591
05/16/2019 00:51:22 - INFO - __main__ -   sent6b: esize=-0.9817726769180993 pval=0.9754858718517502 | w_score=0.10126575641334057 esize=1.0139705989367416
05/16/2019 00:51:22 - INFO - __main__ -   sent7: esize=1.2798993742153038 pval=0.005106458413091977 | w_score=0.09878706406120076 esize=1.2062958643968125
05/16/2019 00:51:22 - INFO - __main__ -   sent7b: esize=-1.295803894299667 pval=0.9952349304530207 | w_score=0.14329603960504755 esize=1.386393532767682
05/16/2019 00:51:22 - INFO - __main__ -   sent8: esize=-0.02474193014134302 pval=0.5208617080228762 | w_score=0.06861062826854834 esize=0.8789026703309939
05/16/2019 00:51:22 - INFO - __main__ -   sent8b: esize=-0.21044128754068606 pval=0.6618878737749775 | w_score=0.014455478580202907 esize=0.18041692510359722

B2 debias:
05/16/2019 00:52:18 - INFO - __main__ -   sent6: esize=-0.8019114607808806 pval=0.9452890327571626 | w_score=0.0009934194386004768 esize=0.6229175819887147
05/16/2019 00:52:18 - INFO - __main__ -   sent6b: esize=-1.253519107615969 pval=0.9939359864929269 | w_score=0.06874343380331993 esize=1.2946306989527896
05/16/2019 00:52:18 - INFO - __main__ -   sent7: esize=-1.328715191959074 pval=0.9960128575692573 | w_score=0.002317296607153861 esize=0.574775169546435
05/16/2019 00:52:18 - INFO - __main__ -   sent7b: esize=-1.227871575505223 pval=0.9928474464465754 | w_score=0.10213194787502289 esize=1.268140149236406
05/16/2019 00:52:18 - INFO - __main__ -   sent8: esize=0.18582644600406875 pval=0.35301309017750004 | w_score=0.0009284274918692666 esize=0.6703268884855057
05/16/2019 00:52:18 - INFO - __main__ -   sent8b: esize=-0.030144658099185286 pval=0.5244656143374344 | w_score=0.001766495406627655 esize=0.03113748235943941

B2 more fine-grained def pairs
05/16/2019 01:48:19 - INFO - __main__ -   sent6: esize=-0.009047210536015036 pval=0.5075154543934522 | w_score=0.007146369133676746 esize=0.9933252762912589
05/16/2019 01:48:19 - INFO - __main__ -   sent6b: esize=-1.2012886898838158 pval=0.9921256632480513 | w_score=0.011553747579455376 esize=1.2406782305498776
05/16/2019 01:48:19 - INFO - __main__ -   sent7: esize=0.2308465211547036 pval=0.32219846456217693 | w_score=0.002122446894645535 esize=0.21463963765199381
05/16/2019 01:48:19 - INFO - __main__ -   sent7b: esize=-0.7428999963801963 pval=0.9317826167578296 | w_score=0.02054482325911522 esize=0.767262168295776
05/16/2019 01:48:19 - INFO - __main__ -   sent8: esize=-0.6264493986582621 pval=0.8952721064312137 | w_score=0.006552151271275102 esize=0.6194637417814516
05/16/2019 01:48:19 - INFO - __main__ -   sent8b: esize=-0.5848303103591797 pval=0.8789841843923419 | w_score=0.0046556368470191956 esize=0.6039939042447161
'''

if __name__ == '__main__':
	main()