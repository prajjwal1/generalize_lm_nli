for dir in /home/nlp/experiments/big_small/roberta_large/epoch*; do
    python3 run_hans.py --model_name_or_path $dir --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 96 --per_device_eval_batch_size 1024 --output_dir=$dir
done
