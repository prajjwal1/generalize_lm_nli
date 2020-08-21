CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path /home/nlp/experiments/meta/main/checkpoint-2048 --do_eval --max_seq_length 80   --per_device_train_batch_size 2  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/main/checkpoint-2048   --per_device_eval_batch_size 4096 --data_dir /home/nlp/data/glue_data/MNLI --task_name mnli --num_train_epochs=1 --max_sample_limit 4096 --step_size=2e-5 --tokenizer_name bert-large-uncased 





