CUDA_VISIBLE_DEVICES=1 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/random_sampling/10_pct_v2   --fp16 --data_pct 0.02

CUDA_VISIBLE_DEVICES=1 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/random_sampling/20_pct_v2   --fp16 --data_pct 0.08

CUDA_VISIBLE_DEVICES=1 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/random_sampling/30_pct_v2   --fp16 --data_pct 0.16

CUDA_VISIBLE_DEVICES=1 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/random_sampling/40_pct_v2   --fp16 --data_pct 0.32

CUDA_VISIBLE_DEVICES=1 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/random_sampling/50_pct_v2   --fp16 --data_pct 0.64

#CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/60_pct_v2   --fp16 --data_pct 0.6

