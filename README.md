# transformers-importance-sampling
Importance Sampling with Transformers

- [x] Train Albert-base on MNLI
- [ ] Extract Embeddings from Albert
- [ ] Clustering Analysis
- [ ] K means sampling from the clustering
- [ ] Re-train Albert-base models

### Results for Albert-base on MNLI
```
eval_loss = 0.4822026441211716
eval_acc = 0.8115337672904801
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
