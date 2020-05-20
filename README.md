# transformers-importance-sampling
Importance Sampling with Transformers

- [x] Train Albert-base on MNLI
- [x] Extract ['CLS'] representations on MNLI from Albert
- [x] Random Sampling results on MNLI and Hans
- [ ] Clustering Analysis
- [ ] Re-train Albert-base models

For code formatting, run
```
make style
```

### Dependencies
- [Pytorch >= 1.5](https://github.com/pytorch/pytorch)
- [Transformers >= 2.9](https://github.com/huggingface/transformers)

### Results for Albert-base on MNLI
```
mnli/mnli-mm eval_acc: 80.11207 / 81.16354
epoch = 3.0
```
The model has been made publicly available at:
```
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/albert-base-v1-mnli")

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/albert-base-v1-mnli")
```
Training was performed with mixed precision. The following command was used:
```
CUDA_VISIBLE_DEVICES=0 python3 run_glue.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/saved_models/   --fp16
```

### Extracting embeddings from Albert
```
CUDA_VISIBLE_DEVICES=0 python3 get_embeddings.py --model_name_or_path prajjwal1/albert-base-v1-mnli --task_name $TASK_NAME --data_dir $GLUE_DIR/$TASK_NAME --max_seq_len 128 --per_gpu_train_batch_size 512 --output_dir /home/nlp/experiments/
```

### Random Sampling Results on MNLI

To run, random sampling tests, run the following command. You have to specify the percentage (`data_pct`). `0.1` means `10%`.
Additionally, also modify the `output_dir` accordingly. By default, models were trained for 3 epochs.

```
CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/10_pct   --fp16 --data_pct 0.1
```

To run evaluation on HANS:
```
$ CUDA_VISIBLE_DEVICES=0 python3 test_hans.py   --model_name_or_path /home/nlp/experiments/albert-base-v1-mnli/  --do_eval   --data_dir /home/nlp/data/hans   --max_seq_length 128  --output_dir /home/nlp/experiments/hans/   --fp16 --task_name HANS --per_gpu_eval_batch_size 4096

$ cd /hans/directory
$ python3 evaluate_heur_output.py /predictions_from_previous_step (output_dir)
```


Results:

| Data Percentage | eval_acc (mnli / mnli-mm) | Hans entailed (lexical overlap, subsequence, constituent)| Hans non=entailed (lexical overlap, subsequence, constituent)
|-----------------|---------------------------|----------------------------------------------------------|
| 10              | 70.99337 /  72.95565      | (0.0196, 0.0146, 0.0118), (0.9776, 0.9858, 0.991)        |
| 20              | 74.51859 /  76.38323      | (0.1626, 0.0956, 0.0866), (0.8396, 0.862, 0.9364)        |
| 30              | 76.48497 /  77.87835      | (0.1046, 0.0866, 0.1164), (0.9012, 0.8958, 0.9098)       |
| 40              | 77.46306 /  78.55980      | (0.276, 0.2222, 0.1364) , (0.7572, 0.775, 0.9138)        |
| 50              | 78.22720 /  78.88527      | (0.0596, 0.044, 0.042)  , (0.9548, 0.9498, 0.98)         |
| 60              | 78.90983 /  79.69894      | (0.0192, 0.021, 0.0184) , (0.984, 0.9786, 0.9926)        |
| 70              | 79.21548 /  79.99735      | (0.0788, 0.0592, 0.056) , (0.9466, 0.934, 0.974)         | 
| 80              | 79.71472 /  80.29902      | (0.1324, 0.159, 0.1278) , (0.9168, 0.8788, 0.9502)       |
| 90              | 79.88792 /  80.66514      | (0.4688, 0.377, 0.4382) , (0.5936, 0.5972, 0.7106)       |
| 100             | 80.11207 /  81.16354      | (0.0562, 0.0552, 0.028), (0.9782, 0.9684, 0.9904)        |
