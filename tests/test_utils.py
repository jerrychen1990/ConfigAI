# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_utils
   Description :
   Author :       chenhao
   date：          2021/5/19
-------------------------------------------------
   Change Activity:
                   2021/5/19:
-------------------------------------------------
"""
import unittest
from config_ai.utils import *


class TestUtils(unittest.TestCase):
    def test_truncate_seq(self):
        seq = list(range(10))
        self.assertEqual([0, 1, 2, 3], truncate_seq(seq, max_len=4, mode="tail", keep_tail=False))
        self.assertEqual([0, 1, 2, 9], truncate_seq(seq, max_len=4, mode="tail", keep_tail=True))
        self.assertEqual([6, 7, 8, 9], truncate_seq(seq, max_len=4, mode="head", keep_head=False))
        self.assertEqual([0, 7, 8, 9], truncate_seq(seq, max_len=4, mode="head", keep_head=True))

        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], truncate_seq(seq, max_len=12, mode="tail", keep_tail=False))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], truncate_seq(seq, max_len=12, mode="tail", keep_tail=True))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], truncate_seq(seq, max_len=12, mode="head", keep_head=False))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], truncate_seq(seq, max_len=12, mode="head", keep_head=True))

    def test_jdump(self):
        val = dict(a=12, b=12.13342352523, c="c")
        val_str = jdumps(val)
        logger.info(val_str)

    def test_read_config(self):
        path = "../examples/config/tf_seq_labeling_example.ini"
        config = read_config(path)
        logger.info(jdumps(config))

    def test_nfold(self):
        data = list(range(22))
        n_folds = nfold(data, 5)
        logger.info(n_folds)

    def test_inverse_dict(self):
        data = {"a": [1, 2, 3], "b": [4], "c": [5, 6]}
        data = inverse_dict(data, overwrite=False)
        logger.info(data)
        self.assertEqual({1: ['a'], 2: ['a'], 3: ['a'], 4: ['b'], 5: ['c'], 6: ['c']}, data)
        data = inverse_dict(data, overwrite=True)
        logger.info(data)
        self.assertEqual({'a': 3, 'b': 4, 'c': 6}, data)

    def test_find_span(self):
        l = [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        spans = find_span(l, 1)
        logger.info(spans)
