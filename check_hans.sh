CUDA_VISIBLE_DEVICES=1 python3 get_heuristic_stats.py
cd hans
CUDA_VISIBLE_DEVICES=1 python3 run_hans.py --model_name_or_path /home/nlp/experiments/aug --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 1024 --output_dir=/home/nlp/experiments/aug --tokenizer_name bert-base-uncased
cd ~/data/glue_data/hans
python3 evaluate_heur_output.py ~/experiments/aug/hans_predictions.txt
cd ~/transformers-importance-sampling
