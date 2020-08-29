python3 train_siamese.py   --model_name bert-base-uncased --task_name MNLI   --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 60   --per_device_train_batch_size 128  --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/siamese/frozen_bert_base/1 --per_device_eval_batch_size 128 --do_train --fp16 --config_name bert-base-uncased --tokenizer_name bert-base-uncased


for val in 1 2 3; do python3 train_siamese.py   --model_name bert-base-uncased --task_name MNLI   --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 60   --per_device_train_batch_size 256  --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/siamese/frozen_bert_base/$((val+1)) --per_device_eval_batch_size 256 --do_train --fp16 --config_name bert-base-uncased --tokenizer_name bert-base-uncased --model_weights_path /home/nlp/experiments/siamese/frozen_bert_base/$val; done


python3 train_siamese.py   --model_name bert-large-uncased --task_name MNLI   --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 60   --per_device_train_batch_size 128  --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/siamese/frozen_bert_large/1 --per_device_eval_batch_size 128 --do_train --fp16 --config_name bert-large-uncased --tokenizer_name bert-large-uncased

for val in 1 2 3; do python3 train_siamese.py   --model_name bert-large-uncased --task_name MNLI   --do_eval   --data_dir /home/nlp/data/glue_data/MNLI/   --max_seq_length 60   --per_device_train_batch_size 128  --learning_rate 2e-5  --num_train_epochs 1.0   --output_dir /home/nlp/experiments/siamese/frozen_bert_large/$((val+1)) --per_device_eval_batch_size 128 --do_train --fp16 --config_name bert-large-uncased --tokenizer_name bert-large-uncased --model_weights_path /home/nlp/experiments/siamese/frozen_bert_large/$val; done

