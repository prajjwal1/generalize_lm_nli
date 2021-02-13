import sys
sys.path.append('..')
from dataclasses import dataclass, field
from typing import Optional

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
#  import nlpaug.augmenter.char as nac
#  import nlpaug.augmenter.sentence as nas

model_id = 'bert_base'
model_path = '/home/nlp/experiments/big_small/bert_base/epoch_4'

config = AutoConfig.from_pretrained(model_path, 
                                    num_labels=3)
#                                    output_attentions=True)

model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                          config=config)

tokenizer = AutoTokenizer.from_pretrained(model_path)

training_args = TrainingArguments(output_dir='/home/nlp/experiments/aug')

#  mnli_easy_data_args = DataTrainingArguments(task_name = 'mnli', 
                                       #  max_seq_length= 32,
                                       #  data_dir = '/home/nlp/cartography/filtered/' + model_id + '_easy_mnli/cartography_confidence_0.01/MNLI')

mnli_hard_data_args = DataTrainingArguments(task_name = 'mnli', 
                                       max_seq_length= 64,
                                       data_dir = '/home/nlp/cartography/filtered/' + model_id + '_hard_mnli/cartography_confidence_0.05/MNLI')

def build_compute_metrics_fn(task_name):
    def compute_metrics_fn(p):
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics('mnli', preds, p.label_ids)
    return compute_metrics_fn

#  mnli_easy_dataset = GlueDataset(mnli_easy_data_args, tokenizer, mode="train")
mnli_hard_dataset = GlueDataset(mnli_hard_data_args, tokenizer, mode="train")

#  mnli_easy_dataset_valid = GlueDataset(mnli_easy_data_args, tokenizer, mode="dev")
#  mnli_hard_dataset_valid = GlueDataset(mnli_hard_data_args, tokenizer, mode="dev")

#  aug = naw.WordEmbsAug(
      #  model_type='fasttext', model_path='/home/nlp/data/'+'wiki-news-300d-1M.vec',
    #  action="insert")
#  #  aug = nac.RandomCharAug(action="substitute")
#  aug = nac.RandomCharAug(action="swap")
#  aug = nac.RandomCharAug(action="delete")
#  aug = nac.RandomCharAug(action="insert")
aug = naw.WordEmbsAug(
        model_type='word2vec',model_path= '/home/nlp/data/'+'GoogleNews-vectors-negative300.bin',
        action="substitute")

#  aug = naw.WordEmbsAug(
        #  model_type='glove',model_path= '/home/nlp/data/'+'glove.6B.300d.txt',
        #  action="insert")

#  aug = naw.ContextualWordEmbsAug(
        #  model_path='roberta-base', action="substitute")
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

trainer = Trainer(model=model,
                 args=training_args,
                 eval_dataset=augmented_dataset,
                 tokenizer=tokenizer,
                 data_collator=default_data_collator,
                 compute_metrics=build_compute_metrics_fn('mnli'))

print(trainer.evaluate())

