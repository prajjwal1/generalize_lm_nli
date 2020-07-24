for seed in 500 750
do
    for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python3 subsampling_mnli.py   --model_name_or_path bert-base-uncased --task_name MNLI --do_train  --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 128  --per_device_train_batch_size 512   --per_device_eval_batch_size 512 --learning_rate 2e-5 --num_train_epochs 3.0   --output_dir /home/nlp/experiments/$p"_pct_"$seed   --fp16 --data_pct $p --seed $seed --overwrite_output_dir
    done
done
