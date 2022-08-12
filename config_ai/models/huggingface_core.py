#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: huggingface_core.py
@time: 2022/8/9 19:00
"""
import logging
import os
from abc import abstractmethod
from typing import List, Dict

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from snippets import ensure_dir_path, jdumps
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, Trainer, \
    DataCollatorForTokenClassification
from transformers import pipeline

from config_ai.models import AIConfigBaseModel
from config_ai.schema import Example, Task
from config_ai.utils import safe_build_data_cls

logger = logging.getLogger(__name__)

_task_map = {
    Task.TEXT_CLS: ("text-classification", DataCollatorWithPadding),
    Task.TEXT_SPAN_CLS: ("token-classification", DataCollatorForTokenClassification)
}


class HuggingfaceBaseModel(AIConfigBaseModel):
    """
    底层包含Neural Network的model
    """

    def __new__(cls, *args, **kwargs):
        cls.hf_task, cls.collator_cls = _task_map[cls.task]
        return super().__new__(cls)

    def __init__(self, config):
        super().__init__(config)
        self.nn_model = None
        self.pretrained_model_name = self.config["nn_model_config"]["pretrained_model_name"]
        self.pretrained_model_path = os.path.join(os.environ.get("PRETRAIN_HOME", ""), self.pretrained_model_name)
        self.max_len = self.task_config["max_len"]
        self._init_tokenizer()

    def _init_tokenizer(self):
        logger.info(f"initializing tokenizer with pretrain_model:{self.pretrained_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)

    @classmethod
    def load_dataset(cls, data_or_path):
        if isinstance(data_or_path, str) and data_or_path.endswith("jsonl"):
            examples = cls.load_examples(data_or_path)
        else:
            examples = data_or_path
        df = pd.DataFrame.from_records([e.dict() for e in examples])
        ds = Dataset.from_pandas(df)
        return ds

    @abstractmethod
    def _train_preprocess(self, examples: List[Dict], truncation=True):
        raise NotImplementedError

    @abstractmethod
    def _predict_preprocess(self, example: Example):
        raise NotImplementedError

    @abstractmethod
    def _predict_postprocess(self, pred):
        raise NotImplementedError

    @ensure_dir_path
    def save(self, path, save_type="json", **kwargs):
        super().save(path=path, save_type=save_type)
        self.save_nn_model(path=path, **kwargs)
        logger.info("save model done")

    def save_nn_model(self, path, **kwargs):
        if self.nn_model:
            nn_model_path = os.path.join(path, "nn_model.pth")
            logger.info(f"saving nn_model to {nn_model_path}")
            torch.save(self.nn_model, nn_model_path)

    @classmethod
    def load(cls, path, load_type="json", load_nn_model=True):
        model = super().load(path=path, load_type=load_type)
        if load_nn_model:
            model.load_nn_model(path=path)
        logger.info("load model done")
        return model

    def load_nn_model(self, path):
        nn_model_path = os.path.join(path, "nn_model.pth")
        logger.info(f"loading nn_model from {nn_model_path}")
        self.nn_model = torch.load(nn_model_path)

    def train(self, train_data, eval_data=None, train_kwargs={}, callbacks={}, **kwargs):
        logger.info("loading datasets...")
        logger.info(train_data)

        _datasets = DatasetDict(train_dataset=self.load_dataset(train_data))
        if eval_data:
            _datasets.update(eval_dataset=self.load_dataset(eval_data))
        _datasets = _datasets.map(self._train_preprocess, batched=True)
        logger.info(f"datasets:{_datasets}")

        logger.info("building trainer")
        training_args = safe_build_data_cls(
            TrainingArguments, train_kwargs
        )
        # logger.info("train dataset features:")
        # logger.info(jdumps(_datasets["train_dataset"].features))

        data_collator = self.collator_cls(tokenizer=self.tokenizer, padding=True)
        trainer = Trainer(
            model=self.nn_model,
            args=training_args,
            **_datasets,
            data_collator=data_collator,
        )
        logger.info("training starts")
        trainer.train()

    def predict(self, data, **kwargs):
        if isinstance(data, str):
            data = self.load_examples(data_path=data)
        data = [self._predict_preprocess(e) for e in data]
        # logger.info(data)
        pred_cls = pipeline(task=self.hf_task, model=self.nn_model, tokenizer=self.tokenizer, device=0)
        # logger.info(pred_cls)
        preds = pred_cls(data)
        # logger.info(preds)

        preds = [self._predict_postprocess(e) for e in preds]
        return preds
