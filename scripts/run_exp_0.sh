CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/seed_250/10_pct   --fp16 --data_pct 0.1 --seed 250

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/seed_250/20_pct   --fp16 --data_pct 0.2 --seed 250

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/seed_250/30_pct   --fp16 --data_pct 0.3 --seed 250

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/seed_250/40_pct   --fp16 --data_pct 0.4 --seed 250

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/seed_250/50_pct   --fp16 --data_pct 0.5 --seed 250

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v1   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/seed_250/60_pct   --fp16 --data_pct 0.6 --seed 250

