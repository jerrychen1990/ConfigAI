# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     layers
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

from config_ai.backend import span_expand

logger = logging.getLogger(__name__)


class MaskSumLayer(tf.keras.layers.Layer):
    """
    将一个序列中对应的mask是1的部分抽取出来求和
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        将一个序列中，对应的mask值为1的部分抽取出来，再add在一起作为序列的embedding。
        通常在分类任务中，需要将一个序列抽取成一个embedding的时候使用
        Args:
            inputs: inputs=[sequence_embedding, mask]
                sequence_embedding:
                    embedding的序列
                    shape:(sample_num,seq_len,embedding_dim)
                    example:[[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19]]],shape=(1,10,1)
                mask:
                    和sequence一样长的，值为0/1的序列。标识哪些embedding需要被抽取出来
                    shape:(sample_num,seq_len)
                    example:[[1,1,0,0,0,0,0,1,0,1]]
                    含义:抽取sequence_embedding中下标为(0,1,7,9)的部分，concat成为一个embedding
        Returns:
            降维之后的序列batch
            shape:(num_sample, embedding_dim)
            example:[[38]]

        """
        sequence_embedding, mask = inputs
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        # print(sequence_embedding.shape)
        # print(mask.shape)
        masked_sequence_embedding = tf.multiply(sequence_embedding, mask)
        embedding = tf.reduce_sum(masked_sequence_embedding, axis=-2)
        return embedding

    def get_config(self):
        base_config = super(MaskSumLayer, self).get_config()
        rs_dict = dict(**base_config)
        return rs_dict


class DenseSoftmaxLayer(tf.keras.layers.Layer):
    """
    将softmax的结果转化成id+prob的形式
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        在softmax的结果中，将label_size的tensor转化成维度为2：（id, prob）的tensor
        Args:
            inputs: inputs=prob_vec
                prob_vec:
                    softmax的结果
                    shape:(sample_num,seq_len,label_size)
                    example:[[[.2,.4,.4], [1.,0.,0.], [.3,.3,.4]]],shape=(1,3,3)
        Returns:
            保留id和prob的序列
            shape:(num_sample, embedding_dim, 2)
            example:[[[1,.4],[0,1.],[2,.4]]]

        """
        prob_vec = inputs
        id_output = tf.cast(tf.argmax(prob_vec, axis=-1), dtype=tf.float32)
        prob_output = tf.reduce_max(prob_vec, axis=-1)
        id_prob_output = tf.stack([id_output, prob_output], axis=-1)
        return id_prob_output

    def get_config(self):
        base_config = super(DenseSoftmaxLayer, self).get_config()
        rs_dict = dict(**base_config)
        return rs_dict


class TokenExtractLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        从sequence_embedding中抽取出spans指定的开头、结尾向量并拼接
        Args:
            inputs: inputs=[sequence_embedding, spans]
                sequence_embedding:
                    embedding的序列
                    shape:(sample_num,seq_len,embedding_dim)
                    example:[[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19]]],shape=(1,10,1)
                spans:
                    要抽取的token序列
                    shape:(sample_num,token_num)
                    example:[[0,2,4,5]]
                    含义:抽取sequence_embedding中下标为(0,2,4,5)的token的embedding
            **kwargs:
        Returns:
            抽取出的token embedding序列
            shape(sample_num, token_num, embedding_dim)
            example:[[1,5,9,11]]
        """
        sequence_embedding, tokens = inputs
        embedding = tf.compat.v1.batch_gather(sequence_embedding, tokens)
        # print(embedding.shape)
        dim = embedding.shape[-2] * embedding.shape[-1]
        # print(dim)
        embedding = tf.reshape(embedding, (-1, dim))

        return embedding

    def get_config(self):
        base_config = super(TokenExtractLayer, self).get_config()
        rs_dict = dict(**base_config)
        return rs_dict


class ExtractLastTokenLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        从sequence_embedding中抽取出spans指定的开头、结尾向量并拼接
        Args:
            inputs: inputs=[sequence_embedding, spans]
                sequence_embedding:
                    embedding的序列
                    shape:(sample_num,seq_len,embedding_dim)
                    example:[[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19]]],shape=(1,10,1)
                token_len:
                    未padding的token的数目
                    shape:(sample_num)
                    example[5]
            **kwargs:
        Returns:
            抽取出最后一个token的embedding
            shape(sample_num, embedding_dim)
            example:[[1,5,9,11]]
        """
        sequence_embedding, token_len = inputs
        token_idxes = tf.expand_dims(token_len - 1, axis=-1)

        embedding = tf.compat.v1.batch_gather(sequence_embedding, token_idxes)
        embedding = tf.squeeze(embedding, axis=-2)
        return embedding

    def get_config(self):
        base_config = super(TokenExtractLayer, self).get_config()
        rs_dict = dict(**base_config)
        return rs_dict


class Token2SpanLayer(tf.keras.layers.Layer):
    def __init__(self, max_span_len, **kwargs):
        super().__init__(**kwargs)
        self.max_span_len = max_span_len

    @tf.function
    def call(self, inputs, **kwargs):
        """
        从sequence_embedding中抽取出spans指定的开头、结尾向量并拼接
        Args:
            inputs: inputs=[sequence_embedding, spans]
                sequence_embedding:
                    embedding的序列
                    shape:(sample_num,seq_len,embedding_dim)
                    example:[[[1],[3],[5],[7],[9]]],shape=(1,5,1)
            **kwargs:
        Returns:
            抽取出的所有可能的span embedding的matrix
            shape(sample_num, seq_len, max_span_len, 2*embedding_dim)
        """
        sequence_embedding = inputs
        ends = span_expand(sequence_embedding, self.max_span_len, contain_sample_dim=True)
        # logger.info("ends's shape")
        # logger.info(ends.shape)
        starts = tf.tile(tf.expand_dims(sequence_embedding, -2), multiples=[1, 1, self.max_span_len, 1])
        # logger.info("start's shape")
        # logger.info(starts.shape)
        span_embedding = tf.concat([starts, ends], axis=-1)

        return span_embedding

    def get_config(self):
        base_config = super(Token2SpanLayer, self).get_config()
        rs_dict = dict(**base_config, max_span_len=self.max_span_len)
        # logger.info(rs_dict)
        return rs_dict


