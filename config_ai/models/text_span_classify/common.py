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
from typing import List, Tuple

import tensorflow as tf

from config_ai.backend import apply_threshold, n_hot2idx_tensor
from config_ai.evaluate import get_tp_fp_fn_set, get_unique_text_span
from config_ai.models.core import AIConfigBaseModel
from config_ai.models.text_classify.common import multi_label2id_vector
from config_ai.schema import TextSpanClassifyExample, TextSpans, Task

logger = logging.getLogger(__name__)


# 将模型预测出的tensor转化成标注策略下的label
def tensor2labels(tensor, multi_label, id2label, threshold=.5) -> List[Tuple[str, float]]:
    # 允许标签重叠的情况下，用阈值确定i
    if multi_label:
        hard_pred_tensor = apply_threshold(tensor, threshold)
        pred_data = []
        for idx, vec in enumerate(hard_pred_tensor):
            prob_list = tensor[idx]
            label_ids = n_hot2idx_tensor(vec).numpy().tolist()
            pred_data.append({(id2label[label_id], prob_list[label_id]) for label_id in label_ids})

    # 不允许标签重叠的情况下，直接取token_id和prob
    else:
        hard_pred_tensor = tf.argmax(tensor, axis=-1).numpy().tolist()
        pred_data = [(id2label[e], tensor[idx][e]) for idx, e in enumerate(hard_pred_tensor)]

    return pred_data


# 将token_label编码成分类的vector形式
def token_label2classify_label_input(target_token_label_sequence, multi_label, label2id):
    if multi_label:
        classify_label_input = [multi_label2id_vector(e, label2id) for e in target_token_label_sequence]
    else:
        classify_label_input = [label2id[e] for e in target_token_label_sequence]

    return classify_label_input


# 将sequence labeling结果输出
def get_text_span_classify_output(examples: List[TextSpanClassifyExample], preds: List[TextSpans]):
    output = []
    for example, pred in zip(examples, preds):
        rs_item = example.dict(exclude_none=True)
        rs_item.update(predict=[e.dict(exclude_none=True) for e in pred])
        if example:
            true_set = set([get_unique_text_span(s) for s in example.text_spans])
            pred_set = set([get_unique_text_span(s) for s in pred])

            tp_set, fp_set, fn_set = get_tp_fp_fn_set(true_set, pred_set)
            rs_item.update(tp_set=tp_set, fp_set=fp_set, fn_set=fn_set)
        output.append(rs_item)
    return output


#
# def mask_o_inputs(input_mask, classify_vals, mask_ele, mask_percent):
#     """
#     《Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition》中的方法，对O的分类结果采样
#     https://arxiv.org/abs/2012.05426
#     Args:
#         input_mask: 原始输入的input_mask
#         classify_vals:分类结果的vector
#         mask_ele:需要有一定概率mask掉（不参与loss计算）的classify结果
#         mask_percent: mask的比率
#     Returns: 修改过后的input_mask
#
#     """
#     assert len(input_mask) == len(classify_vals)
#     for idx, classify_val in enumerate(classify_vals):
#         if classify_val == mask_ele and random.random() <= mask_percent:
#             input_mask[idx] = 0
#     return input_mask
#

class AbstractTextSpanClassifyModelAIConfig(AIConfigBaseModel, ABC):
    task = Task.TEXT_SPAN_CLS
