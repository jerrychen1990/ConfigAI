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
from snippets import flat


from config_ai.models import AIConfigBaseModel
from config_ai.schema import MLMExample


# 将文本 类型结果输出
def get_mlm_output(examples: List[MLMExample],
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


class AbstractMLMClassifyModel(AIConfigBaseModel, ABC):
    example_cls = MLMExample
    labeled_example_cls = MLMExample



