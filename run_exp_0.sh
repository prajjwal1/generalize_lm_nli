CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/10_pct_v2   --fp16 --data_pct 0.1

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/20_pct_v2   --fp16 --data_pct 0.2

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/30_pct_v2   --fp16 --data_pct 0.3

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/40_pct_v2   --fp16 --data_pct 0.4

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/50_pct_v2   --fp16 --data_pct 0.5

CUDA_VISIBLE_DEVICES=0 python3 subsampling_mnli.py   --model_name_or_path albert-base-v2   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/60_pct_v2   --fp16 --data_pct 0.6

