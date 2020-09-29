# transformers-importance-sampling
Importance Sampling with Transformers

- [x] Train Albert-base on MNLI
- [x] Extract ['CLS'] representations on MNLI from Albert
- [x] Random Sampling results on MNLI and Hans
- [x] Clustering Analysis
- [ ] Freezing model

For code formatting, run
```
make style
```

### Dependencies
- [Pytorch >= 1.5](https://github.com/pytorch/pytorch)
- [Transformers >= 2.9](https://github.com/huggingface/transformers)

### 1. Results for Albert-base on MNLI

Albert-base-v1
```
mnli/mnli-mm eval_acc: 80.11207 / 81.16354
epoch = 3.0
```
Albert-base-v1
```
mnli/mnli-mm eval_acc: 84.84971 / 85.51668
epoch = 3.0
```

The model has been made publicly available at:
```
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/albert-base-v1-mnli") # v1 and v2

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/albert-base-v1-mnli") # v1 and v2
```
Training was performed with mixed precision. The following command was used:
```
CUDA_VISIBLE_DEVICES=0 python3 run_glue.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_device_train_batch_size 256  --per_device_eval_batch_size 4096 --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/saved_models/   --fp16
```

You can freeze the base model by passing in `--freeze_base`.

If you're loading a model from the local path, you're required to pass the `tokenizer_name` parameter explicitly.

### 2. Extracting embeddings from Albert
```
CUDA_VISIBLE_DEVICES=0 python3 get_embeddings.py --model_name_or_path prajjwal1/albert-base-v2-mnli --task_name $TASK_NAME --data_dir $GLUE_DIR/$TASK_NAME --max_seq_len 128 --per_device_train_batch_size 512 --output_dir /home/nlp/experiments/
```

### 3. Random Sampling Results on MNLI (with 5 seeds)

To run, random sampling tests, run the following command. You have to specify the percentage (`data_pct`). `0.1` means `10%`.
Additionally, also modify the `output_dir` accordingly. By default, models were trained for 3 epochs.

```
CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/10_pct   --fp16 --data_pct 0.1
```

To run evaluation on HANS:
```
$ CUDA_VISIBLE_DEVICES=0 python3 hans/test_hans.py --task_name hans --model_type albert --do_eval --data_dir $HANS_DIR --model_name_or_path $MODEL_PATH --max_seq_length 128 --output_dir $MODEL_PATH --per_gpu_eval_batch_size 4096

$ cd /hans/directory
$ python3 evaluate_heur_output.py /predictions_from_previous_step (output_dir)
```


### 4. Sampling from clustering
You can create a clustering sklearn object, save its labels and load them. 
There are two modes to perform clustering:
- `data_pct`: this mode will train on elements extracted from clusters equivalent to `len(dataset)*data_pct`
- `num_clusters_elements`: specify how many clusters you want to train on.
- `centroid`

Both should not be used at a time.
If you want cluster object to be saved, pass in `--cluster_only True` flag


If you're initially running this, use the flag `cluster_only`.

```bash
python3 train_clustering.py   --model_name_or_path bert-base-uncased   --task_name MNLI   --do_train   --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 128   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 1.0   --output_dir /home/nlp/experiments/clustering/2_pct   --fp16 --embedding_path /home/nlp/experiments/cls_embeddings_mnli.pth --batch_size 512 --num_clusters 512 --data_pct 0.2 --cluster_only --cluster_output_path /home/nlp/experiments/cluster_output.pth
```
This will save the clustering object in the `cluster_output_path`, so that the same object can be used if you want to choose to run this code again.

After the cluster labels are saved, you can use this `cluster_output_path` will become `cluster_input_path`:
```
CUDA_VISIBLE_DEVICES=0 python3 train_clustering.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/clustering/0   --fp16 --eps 0.2 --min_samples 50 --embedding_path /home/nlp/experiments/cls_embeddings_mnli.pth --data_pct 0.1 --cluster_input_path /home/nlp/experiments/cluster_labels.npy
```


## Meta Learning
Use a batch size of 2 and one GPU for correct evaluation.
```
CUDA_VISIBLE_DEVICES=0 python3 maml_few_shot.py   --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 2  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/1/   --per_device_eval_batch_size 4096 --data_dir /home/nlp/data/glue_data/MNLI --task_name mnli --num_train_epochs=1 --max_sample_limit 2048 --step_size=2e-5 --overwrite_output_dir
```

## Siamese Transformer
```bash
python3 train_siamese.py   --model_name bert-base-uncased   --linear_dim=4096 --task_name $TASK_NAME   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 32   --per_device_train_batch_size 1024  --learning_rate 2e-5  --num_train_epochs 5.0   --output_dir /home/nlp/experiments/siamese   --overwrite_output_dir --per_device_eval_batch_size 1024 --do_train --input_dim 12 --config_name bert-base-uncased --tokenizer_name bert-base-uncased --fp16 --evaluate_during_training
```

## Adapter Networks
```bash
python3 train_adapter.py --model_name_or_path bert-large-uncased --task_name $TASK_NAME   --do_eval   --data_dir /home/nlp/data/glue_data/$TASK_NAME/  --max_seq_length 80   --per_device_train_batch_size 256 --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/adapter_big/epoch_1   --fp16 --per_device_eval_batch_size 128 --do_train --adapter_config pfeiffer --train_adapter
```
