# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     backend
   Description :
   Author :       chenhao
   date：          2021/4/2
-------------------------------------------------
   Change Activity:
                   2021/4/2:
-------------------------------------------------
"""
import logging
import os
import random

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# 对float类型的tensor应用threshold，使其结果转化为0-1之间
def apply_threshold(t: tf.Tensor, threshold=0.5) -> tf.Tensor:
    rs = tf.cast(t, tf.float32) >= threshold
    rs = tf.cast(rs, t.dtype)
    return rs


# 返回$t中值为1的下标组成的tensor
def n_hot2idx_tensor(t: tf.Tensor) -> tf.Tensor:
    t = tf.cast(t, tf.float32)
    rs = tf.where(t == 1.)[:, 0]
    return rs


# 将一个tensor扩展成span
def span_expand(t: tf.Tensor, max_span_len, contain_sample_dim=False) -> tf.Tensor:
    t_list = []
    rank = len(t.shape)
    for i in range(max_span_len):
        if contain_sample_dim:
            tmp = t[:, i:]
            padding = [[0, 0], [0, i]] + [[0, 0]] * (rank - 2)
        else:
            tmp = t[i:]
            padding = [[0, i]] + [[0, 0]] * (rank - 2)
        tmp = tf.pad(tmp, tf.constant(padding), "CONSTANT")
        t_list.append(tmp)

    axis = 2 if contain_sample_dim else 1
    rs = tf.stack(t_list, axis=axis)
    return rs


def set_tf_config():
    logger.info("setting tensorflow config...")

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    logger.info("current devices:")
    logger.info(f"cpus:{cpus}")
    logger.info(f"gpus:{gpus}")
    logger.info("setting gpu memory allow growth...")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("setting soft device placement...")
    tf.config.set_soft_device_placement(True)
    # logger.info("disabling eager mode...")
    # tf.compat.v1.disable_eager_execution()


def set_random_seed(seed):
    logging.info(f"set random seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

