CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path bert-large-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_large/   --per_device_eval_batch_size 2048 --data_dir /home/nlp/data/glue_data --task_list mnli --eval_task_list hans --eval_steps=100 --save_steps=1000 --num_train_epochs=1 --max_sample_limit 2048 --step_size=0.004 --overwrite_output_dir

cp -r /home/nlp/experiments/meta/bert_large /home/nlp/experiments/meta/bert_large_004

CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path bert-large-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_large/   --per_device_eval_batch_size 2048 --data_dir /home/nlp/data/glue_data --task_list mnli --eval_task_list hans --eval_steps=100 --save_steps=1000 --num_train_epochs=1 --max_sample_limit 2048 --step_size=0.0004 --overwrite_output_dir

cp -r /home/nlp/experiments/meta/bert_large /home/nlp/experiments/meta/bert_large_0004

CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_base/   --per_device_eval_batch_size 2048 --data_dir /home/nlp/data/glue_data --task_list mnli --eval_task_list hans --eval_steps=100 --save_steps=1000 --num_train_epochs=1 --max_sample_limit 2048 --step_size=0.4 --overwrite_output_dir

cp -r /home/nlp/experiments/meta/bert_base /home/nlp/experiments/meta/bert_base_4

CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_base/   --per_device_eval_batch_size 2048 --data_dir /home/nlp/data/glue_data --task_list mnli --eval_task_list hans --eval_steps=100 --save_steps=1000 --num_train_epochs=1 --max_sample_limit 2048 --step_size=0.04 --overwrite_output_dir

cp -r /home/nlp/experiments/meta/bert_base /home/nlp/experiments/meta/bert_base_04

CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_base/   --per_device_eval_batch_size 2048 --data_dir /home/nlp/data/glue_data --task_list mnli --eval_task_list hans --eval_steps=100 --save_steps=1000 --num_train_epochs=1 --max_sample_limit 2048 --step_size=0.004 --overwrite_output_dir

cp -r /home/nlp/experiments/meta/bert_base /home/nlp/experiments/meta/bert_base_004

CUDA_VISIBLE_DEVICES=0 python3 reptile_few_shot.py   --model_name_or_path bert-base-uncased  --do_train  --do_eval --max_seq_length 80   --per_device_train_batch_size 1  --learning_rate 2e-5  --output_dir /home/nlp/experiments/meta/bert_base/   --per_device_eval_batch_size 2048 --data_dir /home/nlp/data/glue_data --task_list mnli --eval_task_list hans --eval_steps=100 --save_steps=1000 --num_train_epochs=1 --max_sample_limit 2048 --step_size=0.0004 --overwrite_output_dir

cp -r /home/nlp/experiments/meta/bert_base /home/nlp/experiments/meta/bert_base_0004







