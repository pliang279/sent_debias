import os

def generate_command(device, task_name, model_version, def_pairs_name, debias, epochs=3,
		output_dir=None, resume_model_path=None,
		tune_bert=True, do_train=True,  no_save=False):
	command = '''
	CUDA_VISIBLE_DEVICES={} python run_classifier.py \
	--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/{} \
	--num_train_epochs {} \
	--task_name {} \
	--do_eval \
	--do_lower_case \
	--normalize \
	--def_pairs_name {} '''.format(device, task_name, epochs, task_name.lower(), def_pairs_name)
	if (output_dir):
		command += " --output_dir {} ".format(output_dir)
	else:
		command += " --output_dir ./aaai-results/{}/{} ".format(task_name.lower(), model_version)
	if (tune_bert): command += " --tune_bert "
	if (debias): command += " --debias "
	if (do_train): command += " --do_train "
	if (resume_model_path): command += " --resume_model_path {} ".format(resume_model_path)
	if (no_save): command += " --no_save"
	print(command)
	return command

def fine_tune():
	names = ["news", "reddit", "sst", "pom", "wikitext"]
	# for i in range(5):
	# 	names.append("domain{}".format(i+1))
	# 	names.append("size{}".format(i+1))

	command = generate_command(device=2, task_name="CoLA",
		model_version="biased",
		def_pairs_name="domain1",
		debias=False, 
		resume_model_path="aaai-results/cola/biased",
		epochs=0)
	os.system(command)

def eval_only(task_name="SST-2", device=0):
        command = generate_command(device=device, task_name=task_name,
                model_version="biased",
                def_pairs_name="domain1",
                debias=False,
		do_train=False)
        os.system(command)

def train_classifier_after_debias(task_name="SST-2", device=0):
	def_pairs_name = "domain5"
	command = generate_command(device=device, task_name=task_name,
		model_version="{}_db_after_ft_23epochs".format(def_pairs_name),
		def_pairs_name=def_pairs_name, 
		debias=True,
		tune_bert=False, # fix bert parameters, train classifier layer only
		resume_model_path="aaai-results/{}/{}_db_after_ft_20epochs".format(
			task_name.lower(), def_pairs_name),
		epochs=10)
	os.system(command)

def main():
	train_classifier_after_debias(task_name="SST-2", device=3)
	# eval_only()
	# train_classifier_after_debias("CoLA")

main()
'''
CUDA_VISIBLE_DEVICES=2 python run_classifier.py \
	--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/SST-2 \
	--task_name sst-2 \
	--output_dir ./aaai-results/sst-2 \
	--do_train \
	--do_eval \
	--do_lower_case \
	--normalize \
	--tune_bert \
	--def_pairs_name large_real \
	--debias


CUDA_VISIBLE_DEVICES=1 python run_classifier.py --data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/SST-2 --bert_model bert-base-uncased --task_name sst-2 --output_dir ./tmp/sst-2 --cache_dir ./bert-results/sst-2 --max_seq_length 128 --do_train --do_eval --do_lower_case --train_batch_size 32 --eval_batch_size 8 --learning_rate 2e-5 --num_train_epochs 3 --warmup_proportion 0.1 --def_pairs_name small_real


CUDA_VISIBLE_DEVICES=3 python run_classifier.py \
	--data_dir /media/bighdd7/irene/debias/sent_tasks/glue_data/SST-2 \
	--task_name sst-2 \
	--output_dir ./tmp/test \
	--cache_dir ./bert-results \
	--bert_model bert-base-uncased \
	--max_seq_length 128 \
	--do_train \
	--do_eval \
	--do_lower_case \
	--train_batch_size 32 \
	--eval_batch_size 8 \
	--learning_rate 2e-5 \
	--num_train_epochs 3 \
	--warmup_proportion 0.1 \

	--tune_bert \
	--def_pairs_name large_real \

	--normalize \
	--debias \
	--no_save
	
'''
