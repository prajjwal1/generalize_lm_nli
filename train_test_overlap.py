import os
from tqdm import tqdm
import regex as re
import pickle
import numpy as np
import re
from transformers import AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import GlueDataset
from nbs.train_test_overlap_utils import cosine_similarity
from sentence_transformers import SentenceTransformer

os.environ["WANDB_DISABLED"] = "true"

pattern = re.compile('[\W_]+')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

mnli_data_args = DataTrainingArguments(task_name = 'mnli-mm',
                                       max_seq_length= 96,
                                       data_dir = '/home/nlp/data/glue_data/MNLI')

mnli_train_dataset = GlueDataset(mnli_data_args, tokenizer, mode="train")
mnli_valid_dataset = GlueDataset(mnli_data_args, tokenizer, mode="dev")

mnli_train_sentences = []
mnli_valid_sentences = []

for sent in tqdm(mnli_train_dataset):
    mnli_train_sentences.append(tokenizer.decode(sent.input_ids, skip_special_tokens=True))
    
for sent in tqdm(mnli_valid_dataset):
    mnli_valid_sentences.append(tokenizer.decode(sent.input_ids, skip_special_tokens=True))
    
cosine_dict = []

#  model = SentenceTransformer('paraphrase-distilroberta-base-v1')
#  embeddings_1 = model.encode(mnli_train_sentences, device='cuda')

#  for valid_idx, valid_sent in enumerate(mnli_valid_sentences):
    #  temp_dict = {}
    #  cosine_scores = cosine_similarity(embeddings_1, valid_sent, model)
    #  train_idx = cosine_scores.argmax()
    #  sim_sent = mnli_train_sentences[train_idx]
    #  max_val = cosine_scores[train_idx]
    
for valid_idx, valid_sent in tqdm(enumerate(mnli_valid_sentences)):
    valid_sent = re.sub(r'[^A-Za-z0-9 ]+', '', valid_sent)
    temp_dict = {}
    max_val = 0
    sim_sent = ""
    train_idx = 0
   
    for train_index, train_sent in enumerate(mnli_train_sentences):
        train_sent = re.sub(r'[^A-Za-z0-9 ]+', '', train_sent)
        cosine_score = cosine_similarity(valid_sent.split(), train_sent.split(), None)
        if cosine_score > max_val:
            max_val = cosine_score
            sim_sent = train_sent
            train_idx = train_index
             
    temp_dict['sim_sent'] = sim_sent
    temp_dict['max_val'] = max_val
    temp_dict['train_idx'] = train_idx
    temp_dict['valid_idx'] = valid_idx
    
    cosine_dict.append(temp_dict)
    
    print('Cosine: ', max_val)
    print('Valid: ', mnli_valid_sentences[valid_idx])
    print('Train: ', mnli_train_sentences[train_idx])
    print()

with open("/home/nlp/experiments/train_test_overlap/similarity-mm-lo.bin", "wb") as fp:
    pickle.dump(cosine_dict, fp)
