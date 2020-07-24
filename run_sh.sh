echo "70 pct"
python3 subsampling_mnli.py   --model_name_or_path bert-base-uncased --task_name MNLI --do_train  --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 128  --per_device_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/70_pct   --fp16 --data_pct 0.7
echo "80 pct"
python3 subsampling_mnli.py   --model_name_or_path bert-base-uncased --task_name MNLI --do_train  --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 128  --per_device_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/80_pct   --fp16 --data_pct 0.8
echo "90 pct"
python3 subsampling_mnli.py   --model_name_or_path bert-base-uncased --task_name MNLI --do_train  --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 128  --per_device_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/90_pct   --fp16 --data_pct 0.9
