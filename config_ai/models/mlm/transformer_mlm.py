# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tf_mlm
   Description :
   Author :       chenhao
   date：          2021/8/19
-------------------------------------------------
   Change Activity:
                   2021/8/19:
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
from snippets import log_cost_time, discard_kwarg, load_lines

from config_ai.data_utils import truncate_record
from config_ai.losses import build_classify_loss_layer
from config_ai.metrics import MetricLayer, masked_sparse_categorical_accuracy
from config_ai.models.mlm.common import AbstractMLMClassifyModel
from config_ai.models.tf_core import TFBasedModel
from config_ai.nn_models import get_mlm_model
from config_ai.optimizers import OptimizerFactory
from config_ai.schema import MLMExample, MASK

logger = logging.getLogger(__name__)


class TransformerMLMModel(AbstractMLMClassifyModel, TFBasedModel):

    def _load_config(self, config):
        super()._load_config(config)
        self.max_len = self.task_config["max_len"]
        self.mask_percent = self.task_config.get("mask_percent", 0.15)

    def build_model(self, pretrained_model_path=None, pretrained_model_tag="bert",
                    pos_weight=1., bilstm_dim_list=[], transformer_kwargs={}, h5_file=None):
        with self.get_scope():
            if hasattr(self, 'keep_token_ids'):
                transformer_kwargs.update(keep_tokens=self.keep_token_ids)

            self.nn_model = get_mlm_model(pretrained_model_path, pretrained_model_tag="bert",
                                          transformer_kwargs=transformer_kwargs,
                                          h5_file=h5_file)
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

    def example2feature(self, example: MLMExample) -> Dict:
        feature = self.tokenizer.do_tokenize(text=example.text)
        tokens = feature["tokens"]
        masks = [e for e in enumerate(tokens) if e[1] == MASK]
        feature["masks"] = masks
        if example.masked_tokens:
            assert len(masks) == len(example.masked_tokens)
            feature["masked_tokens"] = [(m[0], t) for m, t in zip(masks, example.masked_tokens)]
        return feature

    def _feature2records(self, idx, feature: Dict, mode: str) -> List[Dict]:
        record = dict(idx=idx, **feature)
        if mode == "train":
            masked_tokens = feature.get("masked_tokens")
            if not masked_tokens:
                token_infos = [e for e in enumerate(feature["tokens"]) if e[1] not in self.tokenizer.special_tokens]
                masked_tokens = random.sample(token_infos, int(len(token_infos) * self.mask_percent))
            token_output = [0] * len(feature["tokens"])
            tokens = copy.copy(feature["tokens"])
            token_ids = copy.copy(feature["token_ids"])

            for idx, token in masked_tokens:
                token_id = self.tokenizer.token2id(token)
                token_output[idx] = token_id
                if tokens[idx] != MASK:
                    r = random.random()
                    if r <= 0.8:
                        t = MASK
                    elif r <= 0.9:
                        t = random.choice(self.tokenizer.vocabs)
                    else:
                        t = token
                    tokens[idx] = t
                    token_ids[idx] = self.tokenizer.token2id(t)

            record.update(token_output=token_output, masked_tokens=masked_tokens, tokens=tokens, token_ids=token_ids)
        truncate_record(record=record, max_len=self.max_len,
                        keys=["token_ids", "segment_ids", "tokens", "token_output"])
        return [record]

    @discard_kwarg
    @log_cost_time
    def _post_predict(self, features, pred_tensors, show_detail=False, threshold=.5) -> List[List[str]]:
        def _tensor2output(feature, pred_tensor):
            # masked_tokens = feature["masked_tokens"]
            # token2char = feature["token2char"]
            # masked_chars = []
            masks = feature["masks"]
            # logger.info(masks)
            # logger.info(pred_tensor.shape)
            pred_tensor = np.argmax(pred_tensor, axis=-1)
            # logger.info(pred_tensor)
            pred_tokens = [self.tokenizer.id2token(e) for e in pred_tensor]
            # logger.info(pred_tokens)

            # logger.info(pred_tensor.shape)
            # logger.info(pred_tensor)
            masked_token_ids = [pred_tensor[e[0]] for e in masks if e[0] < len(pred_tensor)]
            masked_tokens = [self.tokenizer.id2token(i) for i in masked_token_ids]
            return masked_tokens

        preds = [_tensor2output(f, t) for f, t in zip(features, pred_tensors)]
        return preds
