# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_layers
   Description :
   Author :       chenhao
   date：          2021/4/12
-------------------------------------------------
   Change Activity:
                   2021/4/12:
-------------------------------------------------
"""

from config_ai.layers import *

logger = logging.getLogger(__name__)


class TestLayers(tf.test.TestCase):

    def test_mask_sum_layer(self):
        with self.test_session():
            seq_embedding = tf.constant([[[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]]], dtype=tf.float32)
            mask = tf.constant([[1, 1, 0, 0, 0, 0, 0, 1, 0, 1]])
            logger.info(seq_embedding)
            logger.info(mask)
            layer = MaskSumLayer()
            output = layer((seq_embedding, mask))
            logger.info(output)
            self.assertAllEqual(tf.constant([[38]]), output)

    def test_dense_softmax_layer(self):
        with self.test_session():
            prob_vec = tf.constant([[[.2, .4, .4], [1., 0., 0.], [.3, .3, .4]]], dtype=tf.float32)
            logger.info(prob_vec)
            layer = DenseSoftmaxLayer()
            output = layer(prob_vec)
            logger.info(output)
            self.assertAllEqual(tf.constant([[[1, .4], [0, 1.], [2, .4]]]), output)

    def test_token_extract_layer(self):
        with self.test_session():
            seq_embedding = tf.constant([[[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]]], dtype=tf.float32)
            token_index_lis = tf.constant([[0, 2, 3, 4]])
            logger.info(seq_embedding)
            logger.info(token_index_lis)

            layer = TokenExtractLayer()
            output = layer([seq_embedding, token_index_lis])
            logger.info(output)
            self.assertAllEqual(tf.constant([[1, 5, 7, 9]]), output)

    def test_token2span_layer(self):
        with self.test_session():
            seq_embedding = tf.constant([[[1], [3], [5], [7], [9]]], dtype=tf.float32)
            logger.info(seq_embedding)
            layer = Token2SpanLayer(max_span_len=3)
            output = layer(seq_embedding)
            logger.info(output)
            expected = tf.constant([[[[1.0, 1.0], [1.0, 3.0], [1.0, 5.0]], [[3.0, 3.0], [3.0, 5.0], [3.0, 7.0]],
                                     [[5.0, 5.0], [5.0, 7.0], [5.0, 9.0]], [[7.0, 7.0], [7.0, 9.0], [7.0, 0.0]],
                                     [[9.0, 9.0], [9.0, 0.0], [9.0, 0.0]]]])

            self.assertAllEqual(expected, output)

    def test_extract_last_token_layer(self):
        with self.test_session():
            seq_embedding = tf.constant([[[1,2], [3,4], [5,5], [0,0], [0,0]]], dtype=tf.float32)
            logger.info(seq_embedding)
            token_len = tf.constant([2])
            layer = ExtractLastTokenLayer()
            output = layer([seq_embedding, token_len])
            logger.info(output)
            expected = tf.constant([[3,4]])
            self.assertAllEqual(expected, output)
