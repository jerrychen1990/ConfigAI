# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     models
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
import logging
import os

import tensorflow as tf
from bert4keras.models import build_transformer_model
from tensorflow.keras.layers import Input, Add, Embedding, Bidirectional, LSTM
from tensorflow.keras.models import Model, load_model

logger = logging.getLogger(__name__)


def get_sequence_encoder_model(vocab_size: int,
                               pretrained_model_path=None, pretrained_model_tag="bert", transformer_kwargs={},
                               word_embedding_dim=32, bilstm_dim_list=[]):
    """
    得到一个对序列做encoding的模型。一般是下游分类、序列标注等任务的基础
    Args:
        vocab_size: 词表大小
        pretrained_model_path: 预训练模型（如果有）的地址
        pretrained_model_tag: 预训练模型的类型bert
        word_embedding_dim: embedding向量的的维度。使用预训练模型的时候不生效，使用word-embedding的时候生效
        bilstm_dim_list: BILSTM每一层的维度的一般。默认为,则表示使用bilstm。
                    如果不使用预训练模型，必须使用bilstm（否则无法获得context-aware embedding）
        dropout_rate: 最后一层的drop比例
    Returns:一个encoder模型
    输入:input_ids:(num_sample,max_len), segment_ids(num_sample, max_len)
    输出:embedding:(num_sample,max_len,embedding_len)
    """
    token_ids = Input(shape=(None,), dtype=tf.int32, name='token_ids')
    segment_ids = Input(shape=(None,), dtype=tf.int32, name='segment_ids')
    inputs = [token_ids, segment_ids]
    if not pretrained_model_path and not bilstm_dim_list:
        raise Exception("预训练模型(transformer)和bilstm至少要选择一个！否则无法获得context-aware embedding")

    if pretrained_model_path:
        # 加载预训练模型
        config_path = os.path.join(pretrained_model_path, "config.json")
        checkpoint_path = os.path.join(pretrained_model_path, "model.ckpt")
        test_path = f"{checkpoint_path}.index"
        if not os.path.exists(test_path):
            logger.warning(f"ckpt_path:{test_path} not found, will not load pretrain weights!")
            checkpoint_path = None
        else:
            logger.info(f"loading from pretrained weights: {checkpoint_path}")

        pretrained_model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model=pretrained_model_tag,
            **transformer_kwargs
        )
        embedding = pretrained_model(inputs)
    else:
        word_feature = Add(name="add")([token_ids, segment_ids])
        # 初始化token embedding层
        embedding = Embedding(input_dim=vocab_size, output_dim=word_embedding_dim, name="Word-Embedding",
                              trainable=True, mask_zero=True, embeddings_initializer="uniform")(word_feature)
    for bilstm_dim in bilstm_dim_list:
        embedding = Bidirectional(LSTM(units=bilstm_dim, return_sequences=True))(embedding)

    model = Model(inputs=inputs, outputs=embedding, name="sequence_embedding_model")
    return model


def get_mlm_model(pretrained_model_path="", pretrained_model_tag="bert", transformer_kwargs={}, h5_file=None):
    token_ids = Input(shape=(None,), dtype=tf.int32, name='token_ids')
    segment_ids = Input(shape=(None,), dtype=tf.int32, name='segment_ids')
    inputs = [token_ids, segment_ids]

    if h5_file:
        logger.info(f"loading pretrained keras model from h5 file:{h5_file}")
        pretrained_model = load_model(filepath=h5_file, compile=False)
    else:
        config_path = os.path.join(pretrained_model_path, "config.json")
        checkpoint_path = os.path.join(pretrained_model_path, "model.ckpt")
        test_path = f"{checkpoint_path}.index"
        if not os.path.exists(test_path):
            logger.warning(f"ckpt_path:{test_path} not found, will not load pretrain weights!")
            checkpoint_path = None
        else:
            logger.info(f"loading from pretrained weights: {checkpoint_path}")
        pretrained_model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model=pretrained_model_tag,
            with_mlm=True,
            **transformer_kwargs
        )
    output = pretrained_model(inputs)
    return Model(inputs=inputs, outputs=output, name="mlm_model")


def get_unilm_model(pretrained_model_path="", pretrained_model_tag="bert", transformer_kwargs={}, h5_file=None):
    token_ids = Input(shape=(None,), dtype=tf.int32, name='token_ids')
    segment_ids = Input(shape=(None,), dtype=tf.int32, name='segment_ids')
    inputs = [token_ids, segment_ids]

    if h5_file:
        logger.info(f"loading pretrained keras model from h5 file:{h5_file}")
        pretrained_model = load_model(filepath=h5_file, compile=False)
    else:
        config_path = os.path.join(pretrained_model_path, "config.json")
        checkpoint_path = os.path.join(pretrained_model_path, "model.ckpt")
        test_path = f"{checkpoint_path}.index"
        if not os.path.exists(test_path):
            logger.warning(f"ckpt_path:{test_path} not found, will not load pretrain weights!")
            checkpoint_path = None
        else:
            logger.info(f"loading from pretrained weights: {checkpoint_path}")
        pretrained_model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model=pretrained_model_tag,
            application="unilm",
            **transformer_kwargs
        )
    output = pretrained_model(inputs)
    return Model(inputs=inputs, outputs=output, name="unilm_model")
