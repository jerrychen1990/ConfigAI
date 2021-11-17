# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     common
   Description :
   Author :       chenhao
   date：          2021/8/19
-------------------------------------------------
   Change Activity:
                   2021/8/19:
-------------------------------------------------
"""
from abc import ABC
from typing import List, Dict

from config_ai.models import AIConfigBaseModel
# 测评文本分类的结果
from config_ai.schema import MaskedLanguageModelExample

# eval_relation_classify = eval_text_classify
# get_relation_classify_output = get_text_classify_output
from config_ai.utils import flat


class AbstractMLMClassifyModel(AIConfigBaseModel, ABC):
    example_cls = MaskedLanguageModelExample
    labeled_example_cls = MaskedLanguageModelExample


# 将文本 类型结果输出
def get_mlm_output(examples: List[MaskedLanguageModelExample],
                   preds: List[List[str]]) -> List[Dict]:
    rs_list = []

    for example, pred in zip(examples, preds):
        rs_item = example.dict()
        rs_item.update(predict=pred)
        if example.masked_tokens:
            assert len(pred) <= len(example.masked_tokens)
            acc_num = len([e for e in zip(pred, example.masked_tokens) if e[0] == e[1]])
            acc = acc_num / len(example.masked_tokens) if pred else 0.
            rs_item.update(accuracy=acc)

        rs_list.append(rs_item)
    return rs_list


def eval_mlm(masked_tokens_list: List[List[str]], pred_masked_tokens_list: List[List[str]]) -> dict:
    assert len(masked_tokens_list) == len(pred_masked_tokens_list)
    flat_masked_tokens_list = [(id, t) for idx, tokens in enumerate(masked_tokens_list) for t in tokens]
    pred_flat_masked_tokens_list = [(id, t) for idx, tokens in enumerate(pred_masked_tokens_list) for t in tokens]
    token_num = len(pred_flat_masked_tokens_list)
    acc_num = len(set(flat_masked_tokens_list)&set(pred_flat_masked_tokens_list))

    accuracy = acc_num / token_num if token_num else 0.
    return dict(item_num=len(masked_tokens_list), token_num=token_num, accurate_token_num=acc_num, accuracy=accuracy)
