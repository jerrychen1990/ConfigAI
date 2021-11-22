# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     losses
   Description :
   Author :       chenhao
   date：          2021/4/7
-------------------------------------------------
   Change Activity:
                   2021/4/7:
-------------------------------------------------
"""
import logging

import tensorflow as tf
import tensorflow.keras.backend as K
from functools import wraps
from bert4keras.backend import multilabel_categorical_crossentropy
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy, sparse_categorical_crossentropy, \
    kullback_leibler_divergence

_EPSILON = tf.keras.backend.epsilon()
logger = logging.getLogger(__name__)


class LossLayer(tf.keras.layers.Layer):
    def __init__(self, loss_func, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = loss_func

    @tf.function
    def call(self, inputs, **kwargs):
        loss = self.loss_func(*inputs)
        return loss

    def get_config(self):
        config = {
            'loss_func': tf.keras.losses.serialize(self.loss_func)
        }
        base_config = super(LossLayer, self).get_config()
        rs_dict = dict(**base_config)
        rs_dict.update(**config)
        return rs_dict


def add_rdrop_loss(alpha=4):
    def wrapper(loss_func):
        def func(y_true, y_pred1, y_pred2):
            y_pred = (y_pred1 + y_pred2) / 2
            origin_loss = loss_func(y_true, y_pred)
            kld_loss = kullback_leibler_divergence(y_pred1, y_pred2)
            loss = origin_loss + K.mean(kld_loss) * alpha
            return loss

        return func

    return wrapper


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def add_mask(loss_func):
    def func(*args):
        origin_args = args[:-1]
        mask = args[-1]
        mask = K.cast(mask, K.floatx())
        loss = loss_func(*origin_args)
        loss = K.sum(loss * mask) / K.sum(mask)
        return loss

    return func


def build_classify_loss_layer(multi_label, sparse=True, mcs=False, with_mask=False, name="loss_layer",
                              rdrop_alpha=None):
    """
    构建分类任务的loss层
    Args:
        multi_label: 是否多标签分类
        sparse: 是否将用sparse的方式输出（输出id，而不是向量）
        mcs:
        with_mask: 是否对分类结果做mask
        name: loss层的名称
        rdrop_alpha: rdrop的参数
    Returns:一个keras layer,输出loss

    """
    if mcs:
        loss_func = global_pointer_crossentropy
    elif multi_label:
        loss_func = binary_crossentropy
    else:
        loss_func = sparse_categorical_crossentropy if sparse else categorical_crossentropy
    logger.info(f"build loss layer with loss function:{loss_func}")
    if rdrop_alpha:
        logger.info(f"add rdrop loss with alpha:{rdrop_alpha}")
        loss_func = add_rdrop_loss(alpha=rdrop_alpha)(loss_func)
    if with_mask:
        loss_func = add_mask(loss_func)
    loss_layer = LossLayer(loss_func=loss_func, name=name)
    return loss_layer
