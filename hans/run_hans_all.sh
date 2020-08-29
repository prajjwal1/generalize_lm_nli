for dir in /home/nlp/experiments/orthogonal/bert_base_*/; do
    python3 run_hans.py --model_name_or_path $dir --task_name hans --do_eval --data_dir=/home/nlp/data/glue_data/hans --max_seq_length 128 --per_device_eval_batch_size 384 --output_dir=$dir --config_name bert-base-uncased
done
