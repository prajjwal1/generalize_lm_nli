# python3 train_orthogonal.py   --model_name_or_path bert-base-uncased --task_name MNLI --do_eval   --data_dir /home/nlp/data/glue_data/MNLI   --max_seq_length 128   --per_device_train_batch_size 384 --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/debias/epoch_1  --per_device_eval_batch_size 384 --do_train --dataloader_drop_last --evaluate_during_training --eval_steps 50

for epoch in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    do
        python3 train_orthogonal.py   --model_name_or_path bert-base-uncased --task_name MNLI --do_eval   --data_dir /home/nlp/data/glue_data/MNLI   --max_seq_length 128   --per_device_train_batch_size 384 --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/debias/"epoch_"$((epoch+1))  --per_device_eval_batch_size 384 --do_train --model_weights_path /home/nlp/experiments/debias/"epoch_"$epoch --dataloader_drop_last
    done

