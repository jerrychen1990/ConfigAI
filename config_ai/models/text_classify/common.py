# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     common
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
import logging
from abc import ABC
from typing import Set, List, Dict

from config_ai.constants import EMPTY_LABEL
from config_ai.evaluate import label2set, get_tp_fp_fn_set
from config_ai.models.core import AIConfigBaseModel
from config_ai.schema import Task, TextClassifyExample, LabelOrLabels, Label

logger = logging.getLogger(__name__)


class AbstractTextClassifyModel(AIConfigBaseModel, ABC):
    task = Task.TEXT_CLS


def multi_label2id_vector(label_set: Set[str], label2id):
    """
    多标签，将token_label编码成分类的vector形式
    :param label_set:
    :param label2id:
    :return:
    """
    label_num = len(label2id)
    label_vec = [0] * label_num
    for label in label_set:
        label_vec[label2id[label]] = 1
    return label_vec


# 将文本类型编码成分类的vector形式
def get_classify_output(label_list: List[str], label2id: dict, is_sparse):
    if is_sparse:
        label = label_list[0] if label_list else EMPTY_LABEL
        return label2id[label]
    return multi_label2id_vector(set(label_list), label2id)
#
# 将文本 类型结果输出
def get_text_classify_output(examples: List[TextClassifyExample],
                             preds: List[LabelOrLabels]) -> List[Dict]:
    rs_list = []
    assert len(examples) == len(preds)

    for example, pred in zip(examples, preds):
        rs_item = example.dict()
        if isinstance(pred, list):
            rs_item.update(predict=[l.dict(exclude_none=True) for l in pred])
        else:
            rs_item.update(predict=pred.dict(exclude_none=True))

        if example.label:
            true_set = label2set(example.label)
            pred_set = label2set(pred)
            tp_set, fp_set, fn_set = get_tp_fp_fn_set(true_set, pred_set)
            rs_item.update(tp_set=tp_set, fp_set=fp_set, fn_set=fn_set)
        rs_list.append(rs_item)
    return rs_list
