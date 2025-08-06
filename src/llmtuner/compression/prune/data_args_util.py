import sys
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer,BitsAndBytesConfig,default_data_collator
from datasets import load_dataset
import json
import glob
import torch.distributed as dist
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import random
from tqdm import tqdm

def z_score_fromaxis(matrix, axis=1):
    # print(f'matrix in z_score_fromaxis is  {matrix.shape}')
    if matrix.dim() == 1:
        mean = torch.mean(matrix)
        std = torch.std(matrix)
    else:
        mean = torch.mean(matrix, dim=axis, keepdim=True)
        std = torch.std(matrix, dim=axis, keepdim=True)
    standardized_matrix = (matrix - mean) / (std+1e-7)
    return standardized_matrix


def z_score_whole(matrix):
    mean = torch.mean(matrix)
    std = torch.std(matrix)
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix




def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)
    bits: int = field(
        default=2,
        metadata={"help": "How many bits to use."}
    )
    q_group_size: int = field(
        default=128,
        metadata={"help": "Quantization Group Size."}
    )
    quant_type: str = field(
        default="int2-asym",
        metadata={"help": "Quantization data type to use. Should be one of `int2-asym` or `ste-n2f3`."} 
    )
    clip: str = field(
        default=None,
        metadata={"help": "The path of clip cache"}
    )
    cakld_steps: int = field(
        default=10,
        metadata={"help": "How many step to caculate the coefficient of CAKLD."}
    )
    deepspeed_state_config_save: bool = field(
        default = False,
        metadata={"help": "whether to save the optimizer state and deepspeed config."}
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_sample: int, split: str):
        super().__init__()

        with open(data_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        all_dataset = [json.loads(line.strip()) for line in lines]

        sources, targets = zip(*[(s[0][0], f"{s[0][1]}{tokenizer.eos_token}") for s in all_dataset])

        dataset_size = len(sources)
        max_sample = min(max_sample or dataset_size, dataset_size)
        if max_sample < dataset_size:
            indices = random.sample(range(dataset_size), max_sample)
            self.sources, self.targets = [sources[i] for i in indices], [targets[i] for i in indices]
        else:
            self.sources, self.targets = sources, targets 
                 
        split_num = len(self.sources) // 5
        if split == "train":
            self.sources, self.targets = self.sources[split_num:], self.targets[split_num:]
            print(f"Using {len(self.sources)} samples to train")

            # print("Example Data")
            # print("sources: \n", self.sources[0])
            # print("targets: \n", self.targets[0])

        elif split == "eval":
            self.sources, self.targets = self.sources[:split_num], self.targets[:split_num]
            print(f"Using {len(self.sources)} samples to evaluation")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_sample=data_args.max_train_samples, split="train")
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_sample=data_args.max_train_samples, split="eval")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

