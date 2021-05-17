# for val in 0.50 0.75; do python3 run_glue.py   --model_name_or_path /home/nlp/experiments/cartography/roberta_large_easy_pct_025_epoch_2 --task_name MNLI  --do_eval   --data_dir /home/nlp/cartography/filtered/roberta_large_hard/"cartography_confidence_"$val/MNLI  --max_seq_length 90   --per_device_train_batch_size 256 --learning_rate 2e-5  --num_train_epochs 4.0  --output_dir /home/nlp/experiments/cartography/roberta_large_hard/$val/ --fp16 --per_device_eval_batch_size 256 --do_train --overwrite_output_dir; done
python3 run_glue.py   --model_name_or_path hans/bb/bert-base-uncased-MNLI-v1 --task_name MNLI  --do_eval   --data_dir /home/nlp/data/glue_data/MNLI --max_seq_length 90   --per_device_train_batch_size 8 --learning_rate 2e-5  --num_train_epochs 4.0  --output_dir /home/nlp/experiments/check_token_type --fp16 --per_device_eval_batch_size 1 --do_eval
