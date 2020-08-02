for dir in /home/nlp/experiments/debiasing/epoch*; do
    python3 run_hans.py --model_name_or_path $dir --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 8196 --output_dir=$dir --tokenizer_name bert-base-uncased
done
