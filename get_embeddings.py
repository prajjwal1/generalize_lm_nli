from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (HfArgumentParser, TrainingArguments,
                          glue_output_modes, glue_tasks_num_labels)
from transformers.data.data_collator import DefaultDataCollator

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

num_labels = glue_tasks_num_labels[data_args.task_name]
output_mode = glue_output_modes[data_args.task_name]

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
model = (
    AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    .to(device)
    .eval()
)

train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate=True)
data_collator = DefaultDataCollator()
dataloader = DataLoader(
    train_dataset,
    batch_size=training_args.per_gpu_train_batch_size,
    shuffle=False,
    collate_fn=data_collator.collate_batch,
)
print("Extraction of Embeddings in progress")
cls_embeddings = []
for i in tqdm(dataloader):
    inputs = next(iter(dataloader))
    inputs.pop("labels")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    output = model(**inputs)
    cls_embeddings.append(output[0][:, 0, :].cpu().detach().numpy())  # CLS Token representation
    del inputs, output

print("Storing embeddings at ", training_args.output_dir)
torch.save(cls_embeddings, training_args.output_dir + "cls_embeddings_" + data_args.task_name + ".pth")
