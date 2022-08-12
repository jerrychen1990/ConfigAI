#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: test_seq_labeling.py
@time: 2022/8/12 17:24
"""
import unittest

from config_ai.models.text_span_classify.seq_labeling import *

logger = logging.getLogger(__name__)


class TestBackend(unittest.TestCase):

    def test_decode_hf_entities(self):
        pred = [{'entity': 'B_company', 'score': 0.8083094, 'index': 1, 'word': '浙', 'start': 0, 'end': 1},
                {'entity': 'I_company', 'score': 0.9125274, 'index': 2, 'word': '商', 'start': 1, 'end': 2},
                {'entity': 'I_company', 'score': 0.9352316, 'index': 3, 'word': '银', 'start': 2, 'end': 3},
                {'entity': 'I_company', 'score': 0.92638695, 'index': 4, 'word': '行', 'start': 3, 'end': 4},
                {'entity': 'B_name', 'score': 0.81462413, 'index': 10, 'word': '叶', 'start': 9, 'end': 10},
                {'entity': 'I_name', 'score': 0.8697397, 'index': 11, 'word': '老', 'start': 10, 'end': 11},
                {'entity': 'I_name', 'score': 0.8695107, 'index': 12, 'word': '桂', 'start': 11, 'end': 12}]

        text_spans = decode_hf_entities(pred=pred, seq_label_strategy=SeqLabelStrategy.BIO)
        logger.info(text_spans)
