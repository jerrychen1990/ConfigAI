#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     transformer_seq2seq.py
   Author :       chenhao
   time：          2021/12/7 10:05
   Description :
-------------------------------------------------
"""
import logging
import math

import tensorflow as tf
from keras.layers import Multiply
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Lambda
from typing import Dict, List

from config_ai.data_utils import truncate_record, DataManager, records2batches
from config_ai.layers import ExtractLastTokenLayer
from config_ai.losses import build_classify_loss_layer
from config_ai.metrics import masked_sparse_categorical_accuracy, MetricLayer
from config_ai.models.seq2seq.common import AbstractSeq2SeqModel, BeamSearcher
from config_ai.models.tf_core import TFBasedModel
from config_ai.nn_models import get_unilm_model
from config_ai.optimizers import OptimizerFactory
from config_ai.schema import UnionSeq2SeqExample, LabeledSeq2SeqExample, GenText
from config_ai.utils import load_lines, log_cost_time, discard_kwarg

logger = logging.getLogger(__name__)


class TransformerSeq2SeqModel(AbstractSeq2SeqModel, TFBasedModel):
    custom_objects = {
        "ExtractLastTokenLayer": ExtractLastTokenLayer
    }

    def _load_config(self, config):
        super()._load_config(config)
        self.last_token_model = None
        self.max_len = self.task_config["max_len"]
        self.keep_token_file = self.task_config.get("keep_token_file")
        self.keep_token_ids = None
        self.extra_tokens = []
        self.extra_token_ids = []
        self.extra_token_file = self.task_config.get("extra_token_file")
        if self.extra_token_file:
            self.extra_tokens = load_lines(self.extra_token_file)

    def build_model(self, pretrained_model_path=None, pretrained_model_tag="bert", transformer_kwargs={}, h5_file=None,
                    **kwargs):
        with self.get_scope():
            if hasattr(self, 'keep_token_ids'):
                transformer_kwargs.update(keep_tokens=self.keep_token_ids)

            self.nn_model = get_unilm_model(pretrained_model_path, pretrained_model_tag="bert",
                                            transformer_kwargs=transformer_kwargs,
                                            h5_file=h5_file)
        logger.info("nn model's summary:")
        self.nn_model.summary(print_fn=logger.info)
        self._update_model_dict("test", self.nn_model)
        return self.nn_model

    def compile_model(self, optimizer_name: str, optimizer_args: str, **kwargs):
        logger.info(f"compile model with optimizer_name:{optimizer_name}, optimizer_args:{optimizer_args}")

        with self.get_scope():
            input_ids, segment_ids = self.nn_model.inputs[:2]
            prob_vec = self.nn_model(self.nn_model.inputs)
            self.train_model = Model(inputs=self.nn_model.inputs, outputs=prob_vec)

        target_token_ids = Lambda(lambda x: x[:, 1:], name="target_tokens")(input_ids)
        prob_vec = Lambda(lambda x: x[:, :-1], name="prob_vec")(prob_vec)
        loss_mask = Lambda(lambda x: x[:, 1:], name="loss_mask")(segment_ids)

        loss_layer = build_classify_loss_layer(multi_label=False, with_mask=True)
        loss = loss_layer([target_token_ids, prob_vec, loss_mask])
        self.train_model.add_loss(loss)

        accuracy_func = masked_sparse_categorical_accuracy
        metric_layer = MetricLayer(accuracy_func, name="metric_layer")

        accuracy = metric_layer([target_token_ids, prob_vec, loss_mask])

        self.train_model.add_metric(accuracy, aggregation="mean", name="accuracy")
        optimizer = OptimizerFactory.create(optimizer_name, optimizer_args)
        self.train_model.compile(optimizer=optimizer)

        logger.info("training model's summary:")
        self.train_model.summary(print_fn=logger.info)
        self._update_model_dict("train", self.train_model)
        self._build_gen_model()

    def _build_gen_model(self):

        with self.get_scope():
            token_lens = Input(shape=(), dtype=tf.int32, name='token_len')
            inputs = self.nn_model.inputs + [token_lens]
            prob_vec = self.nn_model.output
            extract_last_token_layer = ExtractLastTokenLayer(name="extract_last_token_layer")
            last_prob = extract_last_token_layer([prob_vec, token_lens])

            self.gen_model = Model(inputs=inputs, outputs=last_prob, name="last_token_model")
        logger.info("gen model's summary:")
        self.gen_model.summary(print_fn=logger.info)
        self._update_model_dict("gen", self.gen_model)
        return self.gen_model

    def example2feature(self, example: UnionSeq2SeqExample) -> Dict:
        src_text = example.text + example.extra_text if example.extra_text else example.text
        tgt_txt = example.tgt_text.text if isinstance(example, LabeledSeq2SeqExample) else None
        feature = self.tokenizer.do_tokenize(text=src_text, extra_text=tgt_txt)
        origin_token_len = len(feature["segment_ids"]) - sum(feature["segment_ids"])
        feature.update(origin_token_len=origin_token_len)
        return feature

    def _feature2records(self, idx, feature: Dict, mode: str, only_copy=False) -> List[Dict]:
        record = dict(idx=idx, **feature)
        if mode == "gen":
            record.update(score=0.)
            origin_token_len = record["origin_token_len"]
            record["token_ids"] = record["token_ids"][:origin_token_len]
            record["tokens"] = record["tokens"][:origin_token_len]
            record["segment_ids"] = record["segment_ids"][:origin_token_len]
        record.update(token_len=len(record["tokens"]))
        truncate_record(record=record, max_len=self.max_len,
                        keys=["token_ids", "segment_ids", "tokens"])
        return [record]

    def _gen_infer(self, records, topk, batch_size, only_copy=False):
        dataset_type, dataset_shape = self.get_dataset_info(mode="gen")
        test_batches = records2batches(records, dataset_shape, batch_size)

        logger.info("infering with tf model...")
        pred_tensor_data = []
        for batch in test_batches:
            pred_batch = self.gen_model(batch, training=False)
            pred_tensor_data.extend(pred_batch)

        topk_pred_data = tf.math.top_k(pred_tensor_data, topk)
        topk_prob_data = topk_pred_data.values.numpy().tolist()
        topk_id_data = topk_pred_data.indices.numpy().tolist()

        preds = []
        for token_ids, probs in zip(topk_id_data, topk_prob_data):
            pred = [(i, self.tokenizer.id2token(i), p) for i, p in zip(token_ids, probs)]
            preds.append(pred)
        return preds

    @discard_kwarg
    @log_cost_time
    def _post_infer(self, features, pred_tensors, show_detail=False, threshold=.5) -> List[List[GenText]]:

        def _tensor2output(feature, pred_tensor) -> List[str]:
            token_len = len(feature["tokens"])
            origin_token_len = feature["origin_token_len"]
            pred_tensor = pred_tensor[origin_token_len - 1:token_len - 2]
            pred_token_ids = tf.argmax(pred_tensor, axis=-1).numpy().tolist()
            pred_tokens = [self.tokenizer.id2token(e) for e in pred_token_ids]
            if show_detail:
                logger.info("pred tokens:")
                logger.info(list(zip(pred_token_ids, pred_tokens)))
            # text = self.tokenizer.decode(pred_tokens)
            text = "".join(pred_tokens)
            return [GenText(text=text)]

        preds = [_tensor2output(feature, tensor) for feature, tensor in zip(features, pred_tensors)]
        return preds

    def _infer(self, data_manager: DataManager, batch_size, mode="test", gen_kwargs={},
                 max_pred_num=None, tf_serving_url=None, show_detail=False, **kwargs) -> List[List[GenText]]:

        if mode == "test":
            return super()._infer(data_manager=data_manager, batch_size=batch_size,
                                    show_detail=show_detail, max_pred_num=max_pred_num,
                                    tf_serving_url=tf_serving_url, **kwargs)

            dataset = self._pre_process(data_manager=data_manager, batch_size=batch_size, mode=mode,
                                        max_num=max_pred_num)
            pred_tensors = self._model_infer(dataset=dataset, model=self.nn_model, tf_serving_url=tf_serving_url,
                                               show_detail=show_detail)
            features = data_manager.get_features(max_num=max_pred_num)
            preds = self._post_infer(features=features, pred_tensors=pred_tensors, show_detail=show_detail)
            return preds
        if mode == "gen":
            beam_searcher = BeamSearcher(pred_func=self._gen_infer)
            records = data_manager.get_records(mode=mode, return_generator=False, max_num=max_pred_num)
            preds = beam_searcher.run(records=records, batch_size=batch_size, show_detail=show_detail, **gen_kwargs)
            assert len(preds) == len(records)
            preds = [[GenText(text="".join(item["tokens"][item["origin_token_len"]:-1]),
                              prob=math.exp(item["score"])) for item in pred]
                     for pred in preds]
            return preds
        raise ValueError(f"invalid mode:{mode}")
