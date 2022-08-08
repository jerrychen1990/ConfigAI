# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tf_global_pointer
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

import numpy as np
import tensorflow as tf
from bert4keras.layers import GlobalPointer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from snippets import seq2dict, log_cost_time, load_lines, discard_kwarg

from config_ai.schema import TextSpans, TextSpan, LabeledTextSpanClassifyExample
from config_ai.data_utils import truncate_record
from config_ai.losses import LossLayer, global_pointer_crossentropy
from config_ai.metrics import MetricLayer, global_pointer_f1_score
from config_ai.models.text_span_classify.common import AbstractTextSpanClassifyModelAIConfig, \
    UnionTextSpanClassifyExample
from config_ai.models.tf_core import TFBasedModel
from config_ai.nn_models import get_sequence_encoder_model
from config_ai.optimizers import OptimizerFactory

logger = logging.getLogger(__name__)


class GlobalPointerModel(AbstractTextSpanClassifyModelAIConfig, TFBasedModel):

    def _load_config(self, config):
        super()._load_config(config)
        self.max_len = self.task_config['max_len']
        self.labels = load_lines(self.task_config['label_file_path'])
        self.label2id, self.id2label = seq2dict(self.labels)
        self.label_num = len(self.label2id)

    def build_model(self, pretrained_model_path=None, pretrained_model_tag="bert",
                    bilstm_dim_list=[], head_size=64, pos_weight=1, **kwargs):
        """
        构建模型
        Args:
            head_size: GlobalPointer层的embedding size
            pretrained_model_path: 预训练模型地址
            pretrained_model_tag: 预训练模型类型bert/...
            bilstm_dim_list: 序列encode过程中如果要接bilstm。输入每个bilstm层的dimension
            pos_weight: 正例的权重
            **kwargs:
        Returns:
            nn模型
        """
        with self.get_scope():
            encoder_model = get_sequence_encoder_model(vocab_size=self.vocab_size,
                                                       pretrained_model_path=pretrained_model_path,
                                                       pretrained_model_tag=pretrained_model_tag,
                                                       bilstm_dim_list=bilstm_dim_list, **kwargs)
            sequence_embedding = encoder_model.output
            output = GlobalPointer(self.label_num, head_size)(sequence_embedding)
            output = Lambda(lambda x: x ** pos_weight, name="pos_weight_layer")(output)
            self.nn_model = Model(inputs=encoder_model.inputs, outputs=[output], name="token_classify_model")
        logger.info("nn model's summary:")
        self.nn_model.summary(print_fn=logger.info)
        self._update_model_dict("test", self.nn_model)
        return self.nn_model

    @discard_kwarg
    def compile_model(self, optimizer_name: str, optimizer_args: dict):
        logger.info(f"compile model with optimizer_name:{optimizer_name}, optimizer_args:{optimizer_args}")
        with self.get_scope():
            classify_output = Input(shape=(self.label_num, None, None), dtype=tf.float32, name='classify_output')
            token_ids, segment_ids = self.nn_model.inputs
            output = self.nn_model([token_ids, segment_ids])
            self.train_model = Model(inputs=[token_ids, segment_ids, classify_output], outputs=[output])
        loss_layer = LossLayer(loss_func=global_pointer_crossentropy, name="loss_layer")
        loss = loss_layer([classify_output, output])
        self.train_model.add_loss(loss)

        accuracy_func = global_pointer_f1_score
        metric_layer = MetricLayer(accuracy_func, name="metric_layer")
        metric = metric_layer([classify_output, output])
        self.train_model.add_metric(metric, aggregation="mean", name="global_pointer_f1_score")
        optimizer = OptimizerFactory.create(optimizer_name, optimizer_args)
        self.train_model.compile(optimizer=optimizer)

        logger.info("training model's summary:")
        self.train_model.summary(print_fn=logger.info)
        self._update_model_dict("train", self.train_model)

    def example2feature(self, example: UnionTextSpanClassifyExample) -> Dict:
        feature = self.tokenizer.do_tokenize(text=example.text, store_map=True)
        if isinstance(example, LabeledTextSpanClassifyExample):
            feature.update(text_spans=[e.dict(exclude_none=True) for e in example.text_spans])
        return feature

    def _feature2records(self, idx, feature: Dict, mode: str) -> List[dict]:
        record = dict(idx=idx, **feature)
        if mode == "train":
            text_spans = feature.get("text_spans")
            if text_spans is None:
                raise ValueError(f"not text_spans key found in train mode!")
            text_spans: TextSpans = [TextSpan(**e) for e in text_spans]
            char2token = record["char2token"]
            token_len = len(record["tokens"])
            classify_output = np.zeros(shape=(self.label_num, token_len, token_len))
            for text_span in text_spans:
                label_id = self.label2id[text_span.label]
                token_start = char2token[text_span.span[0]]
                token_end = char2token[text_span.span[1] - 1]
                classify_output[label_id][token_start][token_end] = 1

            record.update(classify_output=classify_output)
        truncate_record(record=record, max_len=self.max_len, keys=["token_ids", "segment_ids", "tokens"])

        return [record]

    @discard_kwarg
    @log_cost_time
    def _post_infer(self, features, pred_tensors, show_detail) -> List[TextSpans]:
        def _tensor2output(feature, pred_tensor) -> TextSpans:
            text_spans = []
            prob_tensor = tf.math.sigmoid(pred_tensor)
            for l, s, e in zip(*np.where(pred_tensor > 0)):
                if e < s:
                    break
                label = self.id2label[l]
                char_start = feature["token2char"][s][0]
                char_end = feature["token2char"][e][1]
                text = feature["text"][char_start:char_end]
                prob = prob_tensor[l][s][e]
                text_span = TextSpan(text=text, label=label, span=(char_start, char_end), prob=prob)
                text_spans.append(text_span)
            return text_spans

        preds = [_tensor2output(f, p) for f, p in zip(features, pred_tensors)]
        return preds
