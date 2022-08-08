#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     common.py
   Author :       chenhao
   time：          2021/12/7 10:04
   Description :
-------------------------------------------------
"""
import copy
import logging
import math
import random
from abc import ABC

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Iterable

from config_ai.models import AIConfigBaseModel
from config_ai.schema import Seq2SeqExample, LabeledSeq2SeqExample, GenText
from config_ai.utils import groupby, log_cost_time

logger = logging.getLogger(__name__)


# 将文本 类型结果输出
def get_seq2seq_output(examples: List[Seq2SeqExample],
                       preds: List[List[GenText]]) -> List[Dict]:
    rs_list = []

    for example, pred in zip(examples, preds):
        rs_item = example.dict(exclude_none=True, exclude_defaults=True)
        rs_item.update(infer=pred)
        rs_list.append(rs_item)
    return rs_list

class BeamSearcher:
    def __init__(self, pred_func, end_token="[SEP]"):
        self.pred_func = pred_func
        self.end_token = end_token
        self.dones = []

    def run(self, records, beam_size, max_step, sample_num, show_detail=False, **kwargs):
        logger.info(f"running beam search with beam_size:{beam_size}, max_step:{max_step}")
        for step in range(max_step):
            logger.info(f"searching step:{step}")
            logger.info(f"current record size:{len(records)}")
            if not records:
                logger.info("no valid record for further generating, break looping")
                break
            preds = self.infer(records=records, topk=beam_size, **kwargs)
            assert len(preds) == len(records)
            records = self.update_records(records, preds)
            records = self.groupby_records(records=records, beam_size=beam_size, show_detail=show_detail)

        results = self.gen_results(records=records, sample_num=sample_num, show_detail=show_detail)
        return results

    def gen_results(self, records, sample_num, show_detail=False):
        logger.info(f"merging {len(records)} max_step records and {len(self.dones)} done records...")
        records += self.dones
        record_dict = groupby(records, key=lambda x: x["idx"])
        rs_records = []
        for idx in sorted(record_dict.keys()):
            items = record_dict[idx]
            items.sort(key=lambda x: (self.get_avg_score(x), x["token_len"]), reverse=True)
            items = items[:sample_num]
            if show_detail:
                self.show_records(idx, items)
            rs_records.append(items)
        return rs_records

    @classmethod
    def get_avg_score(cls, record):
        score = record['score']
        avg_score = score / (record["token_len"] - record["origin_token_len"])
        return avg_score

    def show_records(self, idx, records):
        logger.info(f"records for idx:{idx}")
        for record in records:
            pred_tokens = record["tokens"][record["origin_token_len"]:]
            score = record['score']
            avg_score = self.get_avg_score(record)
            logger.info(
                f"text:{record['text']}, pred_tokens:{pred_tokens}, score:{score:2.3f}, avg_score:{avg_score:2.3f}")

    def groupby_records(self, records, beam_size, show_detail=False):
        logger.info("grouping records")
        record_dict = groupby(records, key=lambda x: x["idx"])
        rs_records = []
        for idx in sorted(record_dict.keys()):
            items = record_dict[idx]
            items.sort(key=lambda x: x["score"], reverse=True)
            items = items[:beam_size]
            if show_detail:
                self.show_records(idx, items)
            rs_records.extend(items)
        return rs_records

    def update_records(self, records, preds):
        logger.info("updating features ...")
        rs_records = []
        for record, pred in zip(records, preds):
            for token_id, token, prob in pred:
                tmp = copy.copy(record)
                tmp["token_ids"] = tmp["token_ids"] + [token_id]
                tmp["tokens"] = tmp["tokens"] + [token]
                tmp["segment_ids"] = tmp["segment_ids"] + [1]
                tmp["score"] += math.log(prob + 1e-7)
                tmp["token_len"] += 1
                if token == self.end_token:
                    self.dones.append(tmp)
                else:
                    rs_records.append(tmp)

        return rs_records

    @log_cost_time
    def infer(self, records, topk, **kwargs) -> Iterable:
        return self.pred_func(records=records, topk=topk, **kwargs)


def random_infer(records, topk) -> Iterable:
    logger.info("infering...")
    preds = []
    for _ in records:
        pred = []
        for _ in range(topk):
            token_id = random.randint(0, 10)
            token = str(token_id)
            prob = random.random()
            pred.append((token_id, token, prob))
        preds.append(pred)
    return preds


RANDOM_BEAM_SEARCHER = BeamSearcher(pred_func=random_infer)


class AbstractSeq2SeqModel(AIConfigBaseModel, ABC):
    example_cls = Seq2SeqExample
    labeled_example_cls = LabeledSeq2SeqExample
