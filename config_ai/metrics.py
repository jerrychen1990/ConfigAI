# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     evaluate_utils
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""

import logging
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import *

logger = logging.getLogger(__name__)


# 将原始的metric计算函数origin_metric_func转化成一个接受mask输入的metric计算函数
def masked_metric(origin_metric_func):
    def wrapper(y_true, y_pred, mask):
        # print("y_true:", y_true)
        # print("y_pred:", y_pred)
        # print("mask:", mask.shape)
        mask = tf.cast(mask, K.floatx())
        origin_metric = origin_metric_func(y_true, y_pred)
        # print("origin_metric", origin_metric)
        metric = tf.reduce_sum(origin_metric * mask) / tf.reduce_sum(mask)
        return metric

    return wrapper


# 接受mask输入的Metric计算层
class MetricLayer(tf.keras.layers.Layer):
    def __init__(self, metric_func, **kwargs):
        super().__init__(**kwargs)
        self.metric_func = metric_func

    @tf.function
    def call(self, inputs, **kwargs):
        metric = self.metric_func(*inputs)
        return metric

    def get_config(self):
        config = {
            'metric_func': tf.keras.losses.serialize(self.metric_func)
        }
        base_config = super(MetricLayer, self).get_config()
        rs_dict = dict(**base_config)
        rs_dict.update(**config)
        return rs_dict


# 带mask的sparse_categorical_accuracy
masked_sparse_categorical_accuracy = masked_metric(sparse_categorical_accuracy)
# 带mask的categorical_accuracy
masked_categorical_accuracy = masked_metric(categorical_accuracy)
# 带mask的binary_accuracy
masked_binary_accuracy = masked_metric(binary_accuracy)


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)
