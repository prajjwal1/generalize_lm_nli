import sys
import os
sys.path.append('..')
import json
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import re
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments, TrainingArguments
from transformers import GlueDataset, default_data_collator, Trainer, glue_compute_metrics

from tqdm import trange

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas

os.environ["WANDB_DISABLED"] = "true"

model_id = 'bert_base'
model_path = '/home/nlp/experiments/big_small/bert_base/epoch_4'
aug_op = 'insert'

config = AutoConfig.from_pretrained(model_path, 
                                    num_labels=3)
#                                    output_attentions=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                          config=config)

tokenizer = AutoTokenizer.from_pretrained(model_path)

training_args = TrainingArguments(output_dir='/home/nlp/experiments/aug', per_device_eval_batch_size=16)
mnli_hard_data_args = DataTrainingArguments(task_name = 'mnli', 
                                       max_seq_length= 96,
                                       data_dir = '/home/nlp/cartography/filtered/' + model_id + '_hard_mnli/cartography_confidence_0.05/MNLI',
                                           overwrite_cache=True)

def build_compute_metrics_fn(task_name):
    def compute_metrics_fn(p):
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics('mnli', preds, p.label_ids)
    return compute_metrics_fn

# mnli_easy_dataset = GlueDataset(mnli_easy_data_args, tokenizer, mode="train")
mnli_hard_dataset = GlueDataset(mnli_hard_data_args, tokenizer, mode="train")

aug = nac.RandomCharAug(action=aug_op)

#  aug = naw.WordEmbsAug(
    #  model_type='word2vec', model_path='/home/nlp/data/'+'GoogleNews-vectors-negative300.bin',
    #  action="substitute", aug_p=0.1)

#  aug = naw.WordEmbsAug(
    #  model_type='fasttext', model_path='/home/nlp/data/'+'wiki-news-300d-1M.vec',
    #  action="substitute", aug_p=0.1)

#  aug = naw.WordEmbsAug(
    #  model_type='glove', model_path='/home/nlp/data/'+'glove.6B.300d.txt',
    #  action="substitute", aug_p=0.1)
#  aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='substitute')
#  aug = naw.SynonymAug(aug_src='wordnet')
#  aug = naw.AntonymAug()

def roberta_augment_dataset(aug, dataset):
    modified_dataset = []
    for i in trange(len(dataset)):
        text = tokenizer.decode(dataset[i].input_ids, skip_special_tokens=False)
        hypothesis = re.search('<s>(.+?)</s>', text).group(1)
        premise = re.search('</s>(.+?)</s>', text).group(1).replace('</s>', '')
        modified_hypothesis = aug.augment(hypothesis)
        modified_premise = aug.augment(premise)
        dict_output = tokenizer(modified_hypothesis, modified_premise, padding='max_length', max_length=128, truncation=True)
        dict_output['label'] = dataset[i].label
        modified_dataset.append(dict_output)
    return modified_dataset


def bert_augment_dataset(aug, dataset):
    modified_dataset = []
    for i in trange(len(dataset)):
        text = tokenizer.decode(dataset[i].input_ids, skip_special_tokens=False)
        hypothesis = re.search('[CLS](.+?)[PAD]', text).group(1).replace('LS] ', '').replace(' [SE', '')
        premise = re.search('[PAD](.+?)[PAD]', text).group(1).replace('] ', '').replace(' [SE', '')
        modified_hypothesis = aug.augment(hypothesis)
        modified_premise = aug.augment(premise)
        dict_output = tokenizer(modified_hypothesis, modified_premise, padding='max_length', max_length=128, truncation=True)
        dict_output['label'] = dataset[i].label
        modified_dataset.append(dict_output)
    return modified_dataset

augmented_dataset = bert_augment_dataset(aug, mnli_hard_dataset)

label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

trainer = Trainer(model=model,
                 args=training_args,
                 eval_dataset=augmented_dataset,
                 tokenizer=tokenizer,
                 data_collator=default_data_collator,
                 compute_metrics=build_compute_metrics_fn('mnli'))

original_preds = trainer.predict(mnli_hard_dataset)

augmented_preds = trainer.predict(augmented_dataset)

flipped_labels = []
all_original_preds = []
all_augmented_preds = []
all_gts = []

for i in trange(len(augmented_dataset)):
    original_pr = np.argmax(original_preds.predictions[i])
    augmented_pr = np.argmax(augmented_preds.predictions[i])
    ground = original_preds.label_ids[i]

    all_original_preds.append(original_pr)
    all_augmented_preds.append(augmented_pr)
    all_gts.append(ground)

    if original_pr != ground:
        if augmented_pr == ground:
            flipped_labels.append(i)

print('Ground: ', Counter(all_gts))
print('Original Preds: ', Counter(all_original_preds))
print('Augmented: ', Counter(all_augmented_preds))

print(len(flipped_labels))
e_n, n_e, e_c, c_e, n_c, c_n = [],[],[],[],[],[]

for i in range(len(flipped_labels)):
    idx = flipped_labels[i]
    aug_sent = tokenizer.decode(augmented_dataset[idx].input_ids, skip_special_tokens=True)
    orig_sent = tokenizer.decode(mnli_hard_dataset[idx].input_ids, skip_special_tokens=True)
    val = {}
    diff = list(set(aug_sent.split())-set(orig_sent.split()))
    val['augmented'] = aug_sent
    for mod_token in diff:
        val['augmented'] = val['augmented'].replace(mod_token, mod_token.upper())
    val['original'] = orig_sent

    if label_dict[np.argmax(original_preds.predictions[idx])] == "entailment" and  label_dict[mnli_hard_dataset[idx].label] == "neutral":
        val['direction'] = "entailment->neutral"
        e_n.append(val)
    if label_dict[np.argmax(original_preds.predictions[idx])] == "neutral" and  label_dict[mnli_hard_dataset[idx].label] == "entailment":
        val['direction'] = "neutral->entailment"
        n_e.append(val)
    if label_dict[np.argmax(original_preds.predictions[idx])] == "entailment" and  label_dict[mnli_hard_dataset[idx].label] == "contradiction":
        val['direction'] = "entailment->contradiction"
        e_c.append(val)
    if label_dict[np.argmax(original_preds.predictions[idx])] == "contradiction" and  label_dict[mnli_hard_dataset[idx].label] == "entailment":
        val['direction'] = "contradiction->entailment"
        c_e.append(val)
    if label_dict[np.argmax(original_preds.predictions[idx])] == "neutral" and  label_dict[mnli_hard_dataset[idx].label] == "contradiction":
        val['direction'] = "neutral->contradiciton"
        n_c.append(val)
    if label_dict[np.argmax(original_preds.predictions[idx])] == "contradiction" and  label_dict[mnli_hard_dataset[idx].label] == "neutral":
        val['direction'] = "contradiction->neutral"
        c_n.append(val)

with open(aug_op + '/e_n.json', 'w') as json_file:
        json.dump(e_n, json_file, indent=4)
with open(aug_op + '/n_e.json', 'w') as json_file:
        json.dump(n_e, json_file, indent=4)
with open(aug_op + '/e_c.json', 'w') as json_file:
        json.dump(e_c, json_file, indent=4)
with open(aug_op + '/c_e.json', 'w') as json_file:
        json.dump(c_e, json_file, indent=4)
with open(aug_op + '/n_c.json', 'w') as json_file:
        json.dump(n_c, json_file, indent=4)
with open(aug_op + '/c_n.json', 'w') as json_file:
        json.dump(c_n, json_file, indent=4)








   #   print(tokenizer.decode(augmented_dataset[idx].input_ids, skip_special_tokens=True))
    #  print(tokenizer.decode(mnli_hard_dataset[idx].input_ids, skip_special_tokens=True))
    #  print(label_dict[np.argmax(original_preds.predictions[idx])], '-> ', label_dict[mnli_hard_dataset[idx].label])
   #   print()
