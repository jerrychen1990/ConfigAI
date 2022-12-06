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

        # text_spans = decode_hf_entities(pred=pred, seq_label_strategy=SeqLabelStrategy.BIO)
        # logger.info(text_spans)

        pred = [{'entity': 'B_organization', 'score': 0.99089384, 'index': 1, 'word': '那', 'start': 0, 'end': 1},
                {'entity': 'I_organization', 'score': 0.99392796, 'index': 2, 'word': '不', 'start': 1, 'end': 2},
                {'entity': 'I_organization', 'score': 0.99457586, 'index': 3, 'word': '勒', 'start': 2, 'end': 3},
                {'entity': 'I_organization', 'score': 0.9925437, 'index': 4, 'word': '斯', 'start': 3, 'end': 4},
                {'entity': 'B_organization', 'score': 0.98314846, 'index': 6, 'word': '锡', 'start': 6, 'end': 7},
                {'entity': 'I_organization', 'score': 0.9948767, 'index': 7, 'word': '耶', 'start': 7, 'end': 8},
                {'entity': 'I_organization', 'score': 0.99288565, 'index': 8, 'word': '纳', 'start': 8, 'end': 9},
                {'entity': 'B_organization', 'score': 0.99251306, 'index': 11, 'word': '桑', 'start': 11, 'end': 12},
                {'entity': 'I_organization', 'score': 0.9936451, 'index': 12, 'word': '普', 'start': 12, 'end': 13},
                {'entity': 'B_organization', 'score': 0.98307633, 'index': 14, 'word': '热', 'start': 15, 'end': 16},
                {'entity': 'I_organization', 'score': 0.9908489, 'index': 15, 'word': '那', 'start': 16, 'end': 17},
                {'entity': 'I_organization', 'score': 0.9921051, 'index': 16, 'word': '亚', 'start': 17, 'end': 18}]

        text_spans = decode_hf_entities(pred=pred, seq_label_strategy=SeqLabelStrategy.BIO)
        logger.info(text_spans)
