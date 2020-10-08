# Generalization in NLI: Ways to [Not] Go Beyond Simple Heuristics
Code for EACL 2021 submission


### Dependencies
- [Pytorch >= 1.5](https://github.com/pytorch/pytorch)
- [Transformers >= 2.9](https://github.com/huggingface/transformers)


The following models were made publicly available during this research:
Pre-Trained LMs: `bert-small`, `bert-tiny`, `bert-mini`, `bert-medium`
Pretrained LMs finetuned on NLI: `albert-base-v2-mnli`, `bert-tiny-mnli`, `bert-small-mnli`, `bert-medium-mnli`, `bert-mini-mnli`, `albert-base-v1-mnli`, `roberta-base-mnli`, `roberta-large-mnli`. To use these, simply use

```
from transformers import AutoModeForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/MODEL_NAME") # v1 and v2
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/MODEL_NAME") # v1 and v2
```
To execute any command, please set the variables
```
export TASK_NAME = MNLI
export GLUE_DIR = /path/to/glue_directory/
```

## Subsampling

### To extract `CLS` embeddings of BERT, use 
```bash
python3 get_embeddings.py --model_name_or_path bert-base-uncased --task_name $TASK_NAME --data_dir $GLUE_DIR/$TASK_NAME --max_seq_len 128 --per_device_train_batch_size 512 --output_dir /home/nlp/experiments/
```

### For random subsampling (increasing data)
Specify the data percentage by `data_pct` argument.

```bash
python3 subsampling_mnli.py   --model_name_or_path bert-large-uncased --task_name   --do_eval  --data_dir /home/nlp/data/glue_data//   --max_seq_length 80  --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 50.0   --output_dir /home/nlp/experiments/overfitting/bert_large  --fp16 --data_pct 0.01 --do_train --evaluate_during_training --eval_steps 8 --overwrite_output_dir --per_device_eval_batch_size 512
```

The following command was also used to studying overfitting aspect in larger models.

### Clustering

For performing clustering, there are multiple ways to go about it.

If you're running clustering for the first time, you need to store clustering results first. Specify the following arguments:
- `batch_size`: Batch size to be used for clustering
- `num_clusters`: How many clusters do you want
- `embedding_path`: Embeddings to be used for performing clustering
- `cluster_output_path`: Where to store clustering output
Optional:
- `cluster_only`: Run only clustering
- `random_state`: For deterministic results with clustering
- `cluster_n_jobs`: Parallel processes to run for clustering

After storing clustering results, you can load them directly with `cluster_input_path`.

Here the arguments relevant to the paper
- `data_pct`: Data percentage to eventually used. Can be used with other strategies.
- `diverse_stream_only` For diverse subsampling, it will extract the most diverse data based on clustering output
- `centroid_elements_only`: Use only the centroids for training

Other arguments:
- `cluster_data_pct`: Use only specified percentage of data from clustering
- `num_clusters_elements`: Use specified number of clusters. 

```bash
python3 train_clustering.py   --model_name_or_path bert-base-uncased  --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUEDIR/$TASK_NAME   --max_seq_length 128   --per_device_train_batch_size 384   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/clustering/diverse_stream/100_pct  --fp16  --use_diverse_stream --data_pct 100 --cluster_input_path /home/nlp/experiments/cluster_output.pth --per_device_eval_batch_size 384 --overwrite_output_dir
```

### HEX with Transformer for debiasing
To use HEX projection debiasing method, use

```bash
python3 train_orthogonal.py   --model_name_or_path bert-base-uncased --task_name $TASK_NAME --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_len 80 --output_dir /home/nlp/experiments/orthogonal/trials/  --dataloader_drop_last --evaluate_during_training --eval_steps 25 --lamb 0.0001 --per_device_train_batch_size 384 --per_device_eval_batch_size 384 --do_train --do_eval
```

### Few-shot (MAML)
If you want to see how bias grows with very less data, use:
- `save_steps`: When to save the model weights
- `eval_steps`: When to perform evaluation
- `max_sample_limit`: How much samples to take

```bash
python3 reptile_few_shot.py   --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 2  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_base/   --per_device_eval_batch_size 4096 --data_dir /home/nlp/data/glue_data/MNLI --task_name mnli --eval_steps=100 --save_steps=1024 --num_train_epochs=1 --max_sample_limit 2048 --step_size=2e-5
```

### Siamese Transformer

```bash
python3 train_siamese.py   --model_name bert-base-uncased --task_name    --do_eval   --data_dir /home/nlp/data/glue_data//   --max_seq_length 60   --per_device_train_batch_size 512  --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/siamese/bert_base --per_device_eval_batch_size 512 --do_train --fp16 --config_name bert-base-uncased --tokenizer_name bert-base-uncased
```

### Adapter Networks
For this, you'll be required to switch from HF `transformers` to [Adapter Transformers](https://github.com/Adapter-Hub/adapter-transformers). This installation may cause issues with rest of the codebase because it relies on older version of `transformers`.
- `train_adapter`: This will freeze the encoder and train only the adapter

```bash
python3 train_adapter.py --model_name_or_path bert-large-uncased --task_name    --do_eval   --data_dir /home/nlp/data/glue_data//   --max_seq_length 80   --per_device_train_batch_size 256 --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/adapter_big/epoch_1   --fp16 --per_device_eval_batch_size 128 --do_train --adapter_config pfeiffer --train_adapter
```

### For running evaluation on HANS

- On any GLUE task
```bash
python3 run_hans.py --model_name_or_path /home/nlp/experiments/clustering/diverse_stream/100_pct --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 1024 --output_dir=/home/nlp/experiments/clustering/diverse_stream/seed7/100_pct/ --tokenizer_name bert-base-uncased
```

In some cases, users are required to pass `model_weights_path` because weights will be loaded in Pytorch like manner (`torch.load`) and not from `AutoModel`

- With Siamese transformers

```bash
python3 run_siamese_hans.py --model_name bert-base-uncased --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 4096 --output_dir=/home/nlp/experiments/siamese/hans --model_weights_path=/home/nlp/experiments/siamese/epoch_4/ --config_name bert-base-uncased --tokenizer_name bert-base-uncased
```

- With HEX

```bash
python3 run_hex_hans.py --model_name_or_path bert-base-uncased --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 384 --output_dir=/home/nlp/experiments/sbert/frozen_bert_base/ --model_weights_path /home/nlp/experiments/$PATH
```

- With Adapter networks

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_hans_adapter.py --model_name_or_path bert-base-uncased --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 4096 --output_dir=/home/nlp/experiments/adapters/epoch_6 --load_task_adapter /home/nlp/experiments/adapters/epoch_6/mnli
```
