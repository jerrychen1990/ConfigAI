# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_tokenizers
   Description :
   Author :       chenhao
   date：          2021/4/21
-------------------------------------------------
   Change Activity:
                   2021/4/21:
-------------------------------------------------
"""
import os
import unittest
from config_ai.tokenizers import *
from config_ai.utils import jdumps

cur_path = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

text = "姚明的身高是2.26米，he is called [TAG] yao in NBA."
sep_idx = 12
expected_tokens = ['[CLS]', '姚', '明', '的', '身', '高', '是', '2', '.', '26', '米', '，', '[SEP]', 'he', 'is', 'call', '##ed', '[TAG]', 'ya',
                   '##o', 'in', 'nba', '.', '[SEP]']

vocab_file = os.path.join(cur_path, "../examples/vocab/roberta_zh_vocab.txt")


class TestTokenizers(unittest.TestCase):
    @unittest.skip("lack data")
    def test_bert4keras_tokenizer(self):
        tokenizer = Bert4kerasTokenizer(vocabs=vocab_file, do_lower_case=True)
        logger.info(f"vocab_size: {tokenizer.vocab_size}")
        tokens = tokenizer.do_tokenize(text, sep_idx)
        logger.info(jdumps(tokens))
        self.assertEqual(expected_tokens, tokens["tokens"])

    @unittest.skip("lack data")
    def test_word_piece_tokenizer(self):
        tokenizer = HFWordPieceTokenizer(vocabs=vocab_file)
        logger.info(f"vocab_size: {tokenizer.vocab_size}")
        logger.info(f"special_tokens: {tokenizer.special_tokens}")
        tokens = tokenizer.do_tokenize(text, sep_idx)
        logger.info(jdumps(tokens))
        self.assertEqual(expected_tokens, tokens["tokens"])

