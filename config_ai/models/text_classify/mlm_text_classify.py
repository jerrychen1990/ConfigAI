#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     mlm_text_classify.py
   Author :       chenhao
   time：          2021/9/22 15:12
   Description :
-------------------------------------------------
"""
import copy
import logging
import random
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from config_ai.data_utils import truncate_record
from config_ai.losses import build_classify_loss_layer
from config_ai.metrics import MetricLayer, masked_sparse_categorical_accuracy
from config_ai.models.text_classify.common import AbstractTextClassifyModel, UnionTextClassifyExample
from config_ai.models.tf_core import TFBasedModel
from config_ai.nn_models import get_mlm_model
from config_ai.optimizers import OptimizerFactory
from config_ai.schema import LabeledTextClassifyExample, Label, LabelOrLabels
from config_ai.utils import log_cost_time, discard_kwarg, jload, inverse_dict, find_span, load_lines, flat

logger = logging.getLogger(__name__)


class MLMTextClassifyModel(AbstractTextClassifyModel, TFBasedModel):

    def __init__(self, config):
        super().__init__(config=config)
        self.tgt_token_ids = [self.tokenizer.token2id(t) for t in self.tgt_tokens]
        self.pred_mask = [1 if idx in self.tgt_token_ids else 0 for idx in range(self.vocab_size)]

    def _load_config(self, config):
        super()._load_config(config)
        self.max_len = self.task_config["max_len"]
        self.word2label = jload(self.task_config['token2label_path'])
        self.label2word = inverse_dict(self.word2label, overwrite=False)
        self.pattern = self.task_config["pattern"]
        # self.keep_tokens = load_lines(self.task_config["keep_token_path"])
        self.tgt_tokens = flat([list(w) for w in self.word2label])
        # self.keep_tokens += self.tgt_tokens
        self.label_num = len(set(self.word2label.values()))

    def build_model(self, pretrained_model_path=None, pretrained_model_tag="bert",
                    pos_weight=1., bilstm_dim_list=[], transformer_kwargs={}, h5_file=None):

        with self.get_scope():
            # transformer_kwargs = {
            #     "keep_tokens": self.keep_token_ids
            # }
            self.nn_model = get_mlm_model(pretrained_model_path, pretrained_model_tag="bert",
                                          transformer_kwargs=transformer_kwargs, h5_file=h5_file)
            logger.info("nn model's summary:")
            self.nn_model.summary(print_fn=logger.info)
            self._update_model_dict("test", self.nn_model)
            return self.nn_model

    @discard_kwarg
    def compile_model(self, optimizer_name, optimizer_args, rdrop_alpha=None):
        logger.info("compiling model...")
        with self.get_scope():
            token_output = Input(shape=(None,), name='token_output', dtype=tf.int32)
            self.train_model = Model(self.nn_model.inputs + [token_output], self.nn_model.output, name="train_model")
        output = self.train_model.output

        loss_mask = Lambda(function=lambda x: tf.cast(tf.not_equal(x, 0), tf.float32), name="pred_mask")(token_output)
        loss_layer = build_classify_loss_layer(multi_label=False, with_mask=True)
        loss = loss_layer([token_output, output, loss_mask])
        self.train_model.add_loss(loss)

        accuracy_func = masked_sparse_categorical_accuracy
        metric_layer = MetricLayer(accuracy_func, name="metric_layer")
        accuracy = metric_layer([token_output, output, loss_mask])

        self.train_model.add_metric(accuracy, aggregation="mean", name="accuracy")
        optimizer = OptimizerFactory.create(optimizer_name, optimizer_args)
        self.train_model.compile(optimizer=optimizer)

        logger.info("training model's summary:")
        self.train_model.summary(print_fn=logger.info)
        self._update_model_dict("train", self.train_model)

    def example2feature(self, example: UnionTextClassifyExample) -> Dict:
        # if example.extra_text:
        #     text = self.pattern
        #     extra_text = self.tokenizer.end_token.join(example.text, example.extra_text)
        # else:
        #     text = self.pattern
        #     extra_text = example.text
        text = self.pattern + example.text
        feature = self.tokenizer.do_tokenize(text=text)

        mask_spans = find_span(feature["tokens"], "[MASK]")
        assert len(mask_spans) == 1
        feature["mask_span"] = mask_spans[0]
        if isinstance(example, LabeledTextClassifyExample):
            if isinstance(example.label, list):
                labels = [e.name for e in example.label]
            else:
                labels = [example.label.name]
            feature.update(labels=labels)
        return feature

    def _feature2records(self, idx, feature: Dict, mode: str) -> List[Dict]:
        record = dict(idx=idx, **feature)
        truncate_record(record=record, max_len=self.max_len, keys=["token_ids", "segment_ids", "tokens"])
        if mode == "train":
            labels = feature.get("labels")
            if labels is None:
                raise ValueError("no labels given in train mode!")
            label = random.choice(labels)
            tgt_word = random.choice(self.label2word[label])
            tokened = self.tokenizer.do_tokenize(tgt_word)
            tgt_token_span = tokened["tokens"][1:-1]
            tgt_token_span_id = tokened["token_ids"][1:-1]
            mask_start, mask_end = record["mask_span"]
            assert len(tgt_token_span) == mask_end - mask_start

            tgt_tokens = copy.copy(record["tokens"])
            tgt_token_ids = copy.copy(record["token_ids"])
            token_output = [0] * len(tgt_token_ids)
            tgt_tokens[mask_start:mask_end] = tgt_token_span
            tgt_token_ids[mask_start:mask_end] = tgt_token_span_id
            token_output[mask_start:mask_end] = tgt_token_span_id
            record.update(target_tokens=tgt_tokens, tgt_token_ids=tgt_token_ids, token_output=token_output)
        return [record]

    @discard_kwarg
    @log_cost_time
    def _post_infer(self, features, pred_tensors, show_detail=False, threshold=.5) -> List[LabelOrLabels]:
        def _tensor2output(feature, pred_tensor) -> LabelOrLabels:
            mask_idx_start, mask_idx_end = feature["mask_span"]
            # logger.info(pred_tensor.shape)
            pred_tensor = pred_tensor[mask_idx_start:mask_idx_end]
            pred_tensor = pred_tensor * self.pred_mask
            # logger.info(pred_tensor)
            # logger.info(pred_tensor.shape)

            probs = np.max(pred_tensor, axis=-1)
            # logger.info(probs)
            prob = np.prod(probs)
            # logger.info(prob)

            pred_tensor = np.argmax(pred_tensor, axis=-1)

            word = "".join(self.tokenizer.id2token(e) for e in pred_tensor)
            # logger.info(word)
            label = self.word2label[word]

            return Label(name=label, prob=prob)

        preds = [_tensor2output(f, t) for f, t in zip(features, pred_tensors)]
        return preds
