#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_evaluate.py
   Author :       chenhao
   time：          2021/11/22 11:14
   Description :
-------------------------------------------------
"""
import json
import logging
import unittest
from ai_schema import *
from ai_schema.eval import eval_text_classify, eval_text_span_classify

logger = logging.getLogger(__name__)


# todo 检查log配置，排查log不输出的问题
class TestEval(unittest.TestCase):
    def test_eval_text_classify(self):
        true_labels = [Label(name="1"), Label(name="2"), Label(name="2"), Label(name="2")]
        pred_labels = [Label(name="1", prob=0.7), Label(name="2"), Label(name="3"), Label(name="1")]
        eval_rs = eval_text_classify(true_labels, pred_labels)
        logger.info(json.dumps(eval_rs, indent=4))
        self.assertEqual(0.5, eval_rs["micro"]["f1"])

    def test_eval_text_span_classify(self):
        true_labels = [
            [TextSpan(label="ORG", text="org1", span=[0, 3]), TextSpan(label="PER", text="per1", span=[0, 3]),
             TextSpan(label="ORG", text="org2", span=[0, 3])],
            [TextSpan(label="ORG", text="org3", span=[0, 3]), TextSpan(label="NUM", text="num1", span=[0, 3])]]
        pred_labels = [
            [TextSpan(label="ORG", text="org1", span=[0, 3]), TextSpan(label="PER", text="per1", span=[0, 2]),
             TextSpan(label="ORG", text="org3", span=[0, 2])],
            [TextSpan(label="NUM", text="num1", span=[0, 3]), TextSpan(label="NUM", text="num2", span=[0, 3])]]
        eval_rs = eval_text_span_classify(true_labels, pred_labels)
        logger.info(json.dumps(eval_rs, indent=4))
        self.assertAlmostEqual(0.4, eval_rs["micro"]["f1"])
