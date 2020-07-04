# Towards Debiasing Sentence Representations

> Pytorch implementation for debiasing sentence representations.

This implementation contains code for removing bias from BERT representations and evaluating bias level in BERT representations.

Correspondence to: 
  - Paul Liang (pliang@cs.cmu.edu)
  - Irene Li (mengzeli@cs.cmu.edu)

## Paper

[**Towards Debiasing Sentence Representations**](https://www.aclweb.org/anthology/2020.acl-main.488/)<br>
[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/), Irene Li, Emily Zheng, Yao Chong Lim, [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), and [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/)<br>
ACL 2020

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
huggingface transformers</br>
numpy 1.18.1</br>
sklearn 0.20.0</br>
matplotlib 3.1.2</br>
gensim 3.8.0 </br>
tqdm 4.45.0</br>
regex 2.5.77</br>
pattern3</br>

The next step is to clone the repository:
```bash
git clone https://github.com/pliang279/sent_debias.git
```

To install bert models, go to `debias-BERT/`, run ```pip install .```

## Data
Download the [GLUE data](https://gluebenchmark.com/tasks) by running this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e):
```python
python download_glue_data.py --data_dir glue_data --tasks SST,QNLI,CoLA
```
Unpack it to some directory `$GLUE_DIR`.

## Precomputed models and embeddings (optional)
1. Models
    * Download https://drive.google.com/file/d/1cAN49-HDHFdNP1GJZn83s2-mEGCeZWjh/view?usp=sharing to `debias-BERT/experiments`.
    * 
      ```
      tar -xvf acl2020-results.tar.gz
      ```

2. Embeddings
    * Download https://drive.google.com/file/d/1ubKn8SCjwnp9pYjQa9SmKWxFojX9a6Bz/view?usp=sharing to `debias-BERT/experiments`.
    * 
      ```
      tar -xvf saved_embs.tar.gz
      ```

## Usage

If you choose to use precomputed models and embeddings, skip to step B. Otherwise, follow step A and B sequentially.

### A. Fine-tune BERT

1. Go to `debias-BERT/experiments`.
2. Run `export TASK_NAME=SST-2` (task can be one of SST-2, CoLA, and QNLI).
4. Fine tune BERT on `$TASK_NAME`.
    * With debiasing
      ```
      python run_classifier.py \
      --data_dir $GLUE_DIR/$TASK_NAME/ \
      --task_name $TASK_NAME \
      --output_dir path/to/results_directory \
      --do_train \
      --do_eval \
      --do_lower_case \
      --debias \
      --normalize \
      --tune_bert 
      ```
    * Without debiasing
      ```
      python run_classifier.py \
      --data_dir $GLUE_DIR/$TASK_NAME/ \
      --task_name $TASK_NAME \
      --output_dir path/to/results_directory \
      --do_train \
      --do_eval \
      --do_lower_case \
      --normalize \
      --tune_bert 
      ```
    The fine-tuned model and dev set evaluation results will be stored under the specified `output_dir`.

### B. Evaluate bias in BERT representations

1. Go to `debias-BERT/experiments`.
2. Run ` export TASK_NAME=SST-2` (task can be one of SST-2, CoLA, and QNLI).
3. Evaluate fine-tuned BERT on bias level.
    * Evaluate debiased fine-tuned BERT.
      ```
        python eval_bias.py \
        --debias \
        --model_path path/to/model \
        --model $TASK_NAME \
        --results_dir path/to/results_directory \
        --output_name debiased
      ```
      If using precomputed models, set `model_path` to `acl2020-results/$TASK_NAME/debiased`.
    * Evaluate biased fine-tuned BERT.
      ```
        python eval_bias.py \
        --model_path path/to/model \
        --model $TASK_NAME \
        --results_dir path/to/results_directory \
        --output_name biased
      ```
      If using precomputed models, set `model_path` to `acl2020-results/$TASK_NAME/biased`.

    The evaluation results will be stored in the file `results_dir/output_name`. 

    Note: The argument `model_path` should be specified as the `output_dir` corresponding to the fine-tuned model you want to evaluate. Specifically, `model_path` should be a directory containing the following files: `config.json`, `pytorch_model.bin` and `vocab.txt`. 
4. Evaluate pretrained BERT on bias level.
    * Evaluate debiased pretrained BERT.
      ```
      python eval_bias.py \
      --debias \
      --model pretrained \
      --results_dir path/to/results_directory \
      --output_name debiased 
      ```
    * Evaluate biased pretrained BERT.
      ```
      python eval_bias.py \
      --model pretrained \
      --results_dir path/to/results_directory \
      --output_name biased 
      ```
    Again, the bias evaluation results will be stored in the file `results_dir/output_name`.
