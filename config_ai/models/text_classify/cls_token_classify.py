# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tf_seq_classify
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""

import logging
from typing import Dict, List

from snippets.utils import seq2dict, load_lines
from transformers import AutoModelForSequenceClassification

from config_ai.models.huggingface_core import HuggingfaceBaseModel
from config_ai.models.text_classify.common import AbstractTextClassifyModel
from config_ai.schema import TextClassifyExample, LabelOrLabels, Label

logger = logging.getLogger(__name__)


class CLSTokenClassifyModel(AbstractTextClassifyModel, HuggingfaceBaseModel):
    auto_model_cls = AutoModelForSequenceClassification

    def _load_config(self, config):
        super()._load_config(config)
        self.multi_label = self.task_config["multi_label"]
        self.max_len = self.task_config["max_len"]
        self.labels = load_lines(self.task_config['label_path'])
        self.sparse_label = not self.multi_label
        self.label2id, self.id2label = seq2dict(self.labels)
        self.label_num = len(self.label2id)

    def build_model(self, **kwargs):
        self.nn_model = self.auto_model_cls.from_pretrained(self.pretrained_model_path, id2label=self.id2label)
        return self.nn_model

    def _train_preprocess(self, examples: List[Dict], truncation=True):
        rs = self.tokenizer(examples["text"], truncation=truncation, max_length=self.max_len)
        rs.update(label=[self.label2id[e["name"]] for e in examples["label"]])

        return rs

    def _predict_preprocess(self, example: TextClassifyExample, show_detail=False):
        return example.text

    def _predict_postprocess(self, pred, show_detail=False) -> LabelOrLabels:
        return Label(name=pred["label"], scoore=pred["score"])
