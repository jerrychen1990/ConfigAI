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
import tensorflow as tf

from typing import Dict, List
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.metrics import (
    binary_accuracy,
    sparse_categorical_accuracy,
)
from tensorflow.keras.models import Model

from config_ai.schema import LabeledTextClassifyExample, Label, LabelOrLabels
from config_ai.backend import apply_threshold, n_hot2idx_tensor
from config_ai.data_utils import truncate_record
from config_ai.losses import build_classify_loss_layer
from config_ai.metrics import MetricLayer
from config_ai.models.text_classify.common import get_classify_output, AbstractTextClassifyModel, UnionTextClassifyExample
from config_ai.models.tf_core import TFBasedModel
from config_ai.nn_models import get_sequence_encoder_model
from config_ai.optimizers import OptimizerFactory
from snippets.utils import seq2dict, load_lines
from snippets.decorators import log_cost_time, discard_kwarg



logger = logging.getLogger(__name__)

class CLSTextClassifyModel(AbstractTextClassifyModel, TFBasedModel):

    def _load_config(self, config):
        super()._load_config(config)
        self.multi_label = self.task_config["multi_label"]
        self.max_len = self.task_config["max_len"]
        self.labels = load_lines(self.task_config['label_path'])
        self.sparse_label = not self.multi_label
        self.label2id, self.id2label = seq2dict(self.labels)
        self.label_num = len(self.label2id)

    def build_model(self, pretrained_model_path=None, pretrained_model_tag="bert",
                    pos_weight=1., bilstm_dim_list=[], transformer_kwargs={}):
        with self.get_scope():
            encoder_model = get_sequence_encoder_model(vocab_size=self.vocab_size,
                                                       pretrained_model_path=pretrained_model_path,
                                                       pretrained_model_tag=pretrained_model_tag,
                                                       bilstm_dim_list=bilstm_dim_list,
                                                       transformer_kwargs=transformer_kwargs)
            sequence_embedding = encoder_model.output
            class_embedding = Lambda(function=lambda x: x[:, 0], name="cls_layer")(sequence_embedding)
            classify_activation = sigmoid if self.multi_label else softmax
            classifier_layer = Dense(
                self.label_num, name="classifier", activation=classify_activation, kernel_initializer=TruncatedNormal(stddev=0.02)
            )
            output = classifier_layer(class_embedding)
            if self.multi_label:
                output = Lambda(lambda x: x ** pos_weight, name="pos_weight_layer")(output)

            self.nn_model = Model(inputs=encoder_model.inputs, outputs=[output], name="nn_model")

        logger.info("nn model's summary:")
        self.nn_model.summary(print_fn=logger.info)
        self._update_model_dict("test", self.nn_model)
        return self.nn_model

    @discard_kwarg
    def compile_model(self, optimizer_name, optimizer_args, rdrop_alpha=None):
        logger.info("compiling model...")
        with self.get_scope():
            classify_output = Input(shape=(self.label_num,) if self.multi_label else (), name='classify_output', dtype=tf.int32)
            inputs = self.nn_model.inputs
            output = self.nn_model(inputs)
            loss_input = [classify_output, output]
            if rdrop_alpha:
                output1 = self.nn_model(inputs)
                loss_input.append(output1)
                output = Lambda(function=lambda x: sum(x) / len(x), name="avg_pool_layer")([output, output1])
            self.train_model = Model(inputs + [classify_output], [output], name="train_model")

        loss_layer = build_classify_loss_layer(multi_label=self.multi_label, rdrop_alpha=rdrop_alpha)
        loss = loss_layer(loss_input)
        self.train_model.add_loss(loss)

        accuracy_func = binary_accuracy if self.multi_label else sparse_categorical_accuracy

        metric_layer = MetricLayer(accuracy_func, name="metric_layer")
        accuracy = metric_layer([classify_output, output])
        self.train_model.add_metric(accuracy, aggregation="mean", name="accuracy")
        optimizer = OptimizerFactory.create(optimizer_name, optimizer_args)
        self.train_model.compile(optimizer=optimizer)

        logger.info("training model's summary:")
        self.train_model.summary(print_fn=logger.info)
        self._update_model_dict("train", self.train_model)

    def _example2feature(self, example: UnionTextClassifyExample) -> Dict:
        feature = self.tokenizer.do_tokenize(text=example.text)
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
            classify_output = get_classify_output(labels, self.label2id, self.sparse_label)
            record.update(classify_output=classify_output)
        return [record]

    @discard_kwarg
    @log_cost_time
    def _post_predict(self, pred_tensors, show_detail=False, threshold=.5) -> List[LabelOrLabels]:
        def _tensor2output(pred_tensor) -> LabelOrLabels:
            if self.multi_label:
                if show_detail:
                    logger.info(f"pred tensor")
                    logger.info(pred_tensor)
                hard_pred_tensor = apply_threshold(pred_tensor, threshold)
                label_data = [int(e.numpy()) for e in n_hot2idx_tensor(hard_pred_tensor)]
                return [Label(name=self.id2label[label_id], prob=pred_tensor[label_id]) for label_id in label_data]
            else:
                label_id = tf.argmax(pred_tensor, axis=-1).numpy()
                label = self.id2label[label_id]
                prob = pred_tensor[label_id]
                return Label(prob=prob, name=label)

        preds = [_tensor2output(t) for t in pred_tensors]
        return preds
