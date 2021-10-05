export TASK_NAME=MNLI; 

python3 new_run_glue.py   --model_name_or_path bert-large-uncased   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_large_ft_47   --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path bert-large-uncased   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_large_ft_59   --overwrite_output_dir --seed 59

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-tiny   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_tiny   --overwrite_output_dir

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-tiny   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_tiny_47   --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-tiny   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_tiny_59   --overwrite_output_dir --seed 59

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-small   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_small   --overwrite_output_dir

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-small   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_small_47   --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-small   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_small_59   --overwrite_output_dir --seed 59

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-medium   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_medium   --overwrite_output_dir

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-medium   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_medium_47   --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-medium   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_medium_59   --overwrite_output_dir --seed 59

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-mini   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_mini  --overwrite_output_dir

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-mini   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_mini_47  --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path prajjwal1/bert-mini   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_bert_mini_59  --overwrite_output_dir --seed 59

python3 new_run_glue.py   --model_name_or_path albert-large-v2   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_albert_large  --overwrite_output_dir

python3 new_run_glue.py   --model_name_or_path albert-large-v2   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_albert_large_47  --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path albert-large-v2   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_albert_large_59  --overwrite_output_dir --seed 59

python3 new_run_glue.py   --model_name_or_path albert-base-v2   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_albert_base  --overwrite_output_dir

python3 new_run_glue.py   --model_name_or_path albert-base-v2   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_albert_base_47  --overwrite_output_dir --seed 47

python3 new_run_glue.py   --model_name_or_path albert-base-v2   --save_strategy epoch   --evaluation_strategy epoch  --task_name MNLI   --do_train   --do_eval   --max_seq_length 96   --per_device_eval_batch_size 256   --per_device_train_batch_size 128   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir ~/experiments/again_albert_base_59  --overwrite_output_dir --seed 59



























