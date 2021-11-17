# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tf_seq_labeling
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

import tensorflow as tf
from ai_schema import LabeledTextSpanClassifyExample, TextSpans, TextSpan
from bert4keras.layers import ConditionalRandomField as CRF
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from config_ai.data_utils import truncate_record
from config_ai.losses import build_classify_loss_layer
from config_ai.metrics import masked_sparse_categorical_accuracy, masked_binary_accuracy, MetricLayer
from config_ai.models.text_span_classify.common import SeqLabelStrategy, read_seq_label_file, \
    get_overlap_token_label_sequence, \
    get_token_label_sequence, token_label2classify_label_input, AbstractTextSpanClassifyModelAIConfig, \
    tensor2labels, decode_overlap_label_sequence, decode_label_sequence, UnionTextSpanClassifyExample
from config_ai.models.tf_core import TFBasedModel
from config_ai.nn_models import get_sequence_encoder_model
from config_ai.optimizers import OptimizerFactory
from config_ai.utils import seq2dict, log_cost_time, discard_kwarg

logger = logging.getLogger(__name__)


class TFSeqLabelingModel(AbstractTextSpanClassifyModelAIConfig, TFBasedModel):

    def _load_config(self, config):
        super()._load_config(config)
        self.seq_label_strategy: SeqLabelStrategy = SeqLabelStrategy[
            self.task_config['seq_label_strategy']]

        self.max_len = self.task_config['max_len']
        self.multi_label = self.task_config.get("multi_label", False)
        self.label_list = read_seq_label_file(self.task_config['label_file_path'], self.seq_label_strategy)
        self.label2id, self.id2label = seq2dict(self.label_list)
        self.label_num = len(self.label2id)

    def build_model(self, pretrained_model_path=None, pretrained_model_tag="bert",
                    bilstm_dim_list=[], use_crf=False, crf_lr_multiplier=100, pos_weight=1, **kwargs):
        """
        构建模型
        Args:
            pretrained_model_path: 预训练模型地址
            pretrained_model_tag: 预训练模型类型bert/...
            dense_dim_list: 序列encode之后过的每个全连接层的维度（默认用relu做激活函数）。如果为空列表，表示不添加全连接层
            hidden_dropout_prob: 序列encode之后过得dropout层drop概率。避免过拟合
            bilstm_dim_list: 序列encode过程中如果要接bilstm。输入每个bilstm层的dimension
            use_crf: 是否使用crf层
            crf_lr_multiplier: crf层的学习率倍数，参考https://kexue.fm/archives/7196
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
            classify_activation = sigmoid if self.multi_label else softmax
            classifier_layer = Dense(
                self.label_num, name="token_classifier", activation=classify_activation,
                kernel_initializer=TruncatedNormal(stddev=0.02)
            )
            prob_vec_output = classifier_layer(sequence_embedding)
            if use_crf:
                classifier_layer = CRF(lr_multiplier=crf_lr_multiplier, name="crf_layer")
                prob_vec_output = classifier_layer(prob_vec_output)
            if self.multi_label:
                prob_vec_output = Lambda(lambda x: x ** pos_weight, name="pos_weight_layer")(prob_vec_output)

            self.nn_model = Model(inputs=encoder_model.inputs, outputs=[prob_vec_output], name="token_classify_model")
        logger.info("nn model's summary:")
        self.nn_model.summary(print_fn=logger.info)
        self._update_model_dict("test", self.nn_model)
        return self.nn_model

    def compile_model(self, optimizer_name: str, optimizer_args: dict, **kwargs):
        logger.info(f"compile model with optimizer_name:{optimizer_name}, optimizer_args:{optimizer_args}")
        with self.get_scope():
            classify_labels = Input(shape=(None, self.label_num) if self.multi_label else (None,),
                                    name='classify_labels', dtype=tf.int32)
            token_ids, segment_ids = self.nn_model.inputs
            output = self.nn_model([token_ids, segment_ids])
            self.train_model = Model(inputs=[token_ids, segment_ids, classify_labels], outputs=[output])

        loss_mask = Lambda(function=lambda x: tf.cast(tf.not_equal(x, 0), tf.float32), name="pred_mask")(token_ids)

        # 计算loss的时候，过滤掉pad token的loss
        loss_layer = build_classify_loss_layer(multi_label=self.multi_label)
        loss = loss_layer([classify_labels, output, loss_mask])
        self.train_model.add_loss(loss)

        # 计算accuracy的时候，过滤掉pad token 的accuracy
        masked_accuracy_func = masked_binary_accuracy if self.multi_label else masked_sparse_categorical_accuracy
        metric_layer = MetricLayer(masked_accuracy_func)
        masked_accuracy = metric_layer([classify_labels, output, loss_mask])
        self.train_model.add_metric(masked_accuracy, aggregation="mean", name="accuracy")

        optimizer = OptimizerFactory.create(optimizer_name, optimizer_args)
        self.train_model.compile(optimizer=optimizer)
        logger.info("training model's summary:")
        self.train_model.summary(print_fn=logger.info)
        self._update_model_dict("train", self.train_model)

    def _example2feature(self, example: UnionTextSpanClassifyExample) -> Dict:
        feature = self.tokenizer.do_tokenize(text=example.text, extra_text=example.extra_text, store_map=True)
        if isinstance(example, LabeledTextSpanClassifyExample):
            feature.update(text_spans=[e.dict() for e in example.label])
        return feature

    def _feature2records(self, idx, feature: Dict, mode: str) -> List[Dict]:
        record = dict(idx=idx, **feature)
        if mode == "train":
            text_spans = feature.get("text_spans")
            if text_spans is None:
                raise ValueError(f"not text_spans key found in train mode!")
            text_spans = [TextSpan(**e) for e in text_spans]
            token_label_func = get_overlap_token_label_sequence if self.multi_label else get_token_label_sequence
            target_token_label_sequence = token_label_func(feature["tokens"], text_spans,
                                                           feature["char2token"], self.seq_label_strategy)
            classify_labels = token_label2classify_label_input(target_token_label_sequence, self.multi_label,
                                                               self.label2id)
            record.update(target_token_label_sequence=target_token_label_sequence, classify_labels=classify_labels)

        truncate_record(record=record, max_len=self.max_len,
                        keys=["token_ids", "segment_ids", "tokens", "classify_labels"])

        return [record]

    @discard_kwarg
    @log_cost_time
    def _post_predict(self, features, pred_tensors, show_detail, threshold=0.5) -> List[TextSpans]:
        def _tensor2output(feature, pred_tensor) -> TextSpans:
            pred_labels = tensor2labels(pred_tensor, self.multi_label, self.id2label, threshold=threshold)
            tokens = feature["tokens"]
            pred_labels = pred_labels[:len(tokens)]
            if show_detail:
                logger.info(f"tokens:{tokens}")
                for idx, (token, pred_label) in enumerate(zip(tokens, pred_labels)):
                    if pred_label and pred_label != self.seq_label_strategy.empty:
                        logger.info(f"idx:{idx}, token:{token}, pred:{pred_label}")
            decode_func = decode_overlap_label_sequence if self.multi_label else decode_label_sequence
            text_spans = decode_func(feature, pred_labels, self.seq_label_strategy)
            return text_spans

        preds = [_tensor2output(f, p) for f, p in zip(features, pred_tensors)]
        return preds
