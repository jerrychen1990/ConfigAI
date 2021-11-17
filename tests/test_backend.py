# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_backend
   Description :
   Author :       chenhao
   date：          2021/4/17
-------------------------------------------------
   Change Activity:
                   2021/4/17:
-------------------------------------------------
"""
from config_ai.backend import *

logger = logging.getLogger(__name__)


class TestBackend(tf.test.TestCase):

    def test_span_expand(self):
        with self.test_session():
            mask = tf.constant([1, 1, 1, 0, 0], dtype=tf.float32)
            logger.info(mask)
            rs = span_expand(mask, 3)
            logger.info(rs.numpy().tolist())
            self.assertAllEqual(tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), rs)
