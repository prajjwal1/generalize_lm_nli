for val in 10 20 30 40 50 60 70 80 90 100; do python3 train_clustering.py   --model_name_or_path bert-base-uncased  --task_name  MNLI  --do_train   --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 80   --per_device_train_batch_size 512   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /home/nlp/experiments/clustering/diverse_stream/seed48/$val"_pct"  --fp16  --use_diverse_stream --data_pct $val --cluster_input_path /home/nlp/experiments/cluster_output.pth --per_device_eval_batch_size 512 --seed 48; done
