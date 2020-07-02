Obtain BERT Downstream Data:
 run 'python download_glue_data.py'

Bias Evaluation Data should already be on github 

Install packages
import numpy, sklearn, pattern3, scipy, gensim, torch, tqdm, regex

Go into pytorch-pretrained-BERT/, run 'pip install .'

---------------------------------------------------------------------------
Fine-tune BERT and evaluate on downstream tasks (SST-2, CoLA, QNLI)
1. Go to pytorch-pretrained-BERT/examples
2. export TASK_NAME=SST-2 (or CoLA, QNLI)
3. Debiased fine-tune:
python run_classifier.py \
	--data_dir ../../glue_data/$TASK_NAME/ \
	--task_name $TASK_NAME \
	--output_dir acl2020-results/$TASK_NAME/debiased_test \
	--do_train \
	--do_eval \
	--do_lower_case \
	--debias \
	--normalize \
	--tune_bert 
4. Biased fine-tune:
python run_classifier.py \
	--data_dir ../../glue_data/$TASK_NAME/ \
	--task_name $TASK_NAME \
	--output_dir acl2020-results/$TASK_NAME/biased \
	--do_train \
	--do_eval \
	--do_lower_case \
	--normalize \
	--tune_bert 
Results will be stored under output_dir.
---------------------------------------------------------------------------

Evaluate Bias Removal
1. Go to pytorch-pretrained-BERT/examples
2. export TASK_NAME=SST-2 (or CoLA, QNLI)
3. Evaluate debiased fine-tuned model
python eval_bias.py \
	--model_path acl2020-results/$TASK_NAME/debiased/ \
	--debias \
	--model $TASK_NAME \
	--output_name debiased
4. Evaluate biased fine-tuned model
python eval_bias.py \
	--model_path acl2020-results/$TASK_NAME/biased/ \
	--model $TASK_NAME \
	--output_name biased
5. Biased pretrained bert
biased: python eval_bias.py --model pretrained --output_name biased --results_dir test06
debiased: python eval_bias.py --model pretrained --output_name debiased --debias

Results will be stored under pytorch-pretrained-BERT/examples/acl2020_bias_eval_results/$model/$output_name