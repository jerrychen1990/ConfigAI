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
from enum import Enum, unique
from typing import List, Tuple, Set, Dict

import tensorflow as tf
from ai_schema import TextSpanClassifyExample, LabeledTextSpanClassifyExample, TextSpan, TextSpans, UnionTextSpanClassifyExample
from ai_schema.eval import get_tp_fp_fn_set

from config_ai.backend import apply_threshold, n_hot2idx_tensor
from config_ai.models.core import AIConfigBaseModel
from config_ai.models.text_classify.common import multi_label2id_vector
from config_ai.utils import load_lines

logger = logging.getLogger(__name__)
LABEL_SEP = "_"


@unique
# 所有的序列标注编码、解码格式BIO/BIOES等等
class SeqLabelStrategy(Enum):
    # BIO编码
    BIO = ("B", "I", None, None, "O", False, False)
    # BIO编码，只有B后面带有span的类型，I不区分span类别
    BIO_SIMPLE = ("B", "I", None, None, "O", False, True)
    # BIOES编码
    BIOES = ("B", "I", "E", "S", "O", False, False)
    # BIOES编码, 只有B和S后面带有span的类型，IEO不区分span的类别
    BIOES_SIMPLE = ("B", "I", "E", "S", "O", False, True)
    # 用B和E两个指正标注span的范围
    BE_POINTER = ("B", None, "E", None, None, True, False)

    def __init__(self, begin, mid, end, single, empty, is_pointer=False, simple_mode=False):
        self.begin = begin
        self.mid = mid
        self.end = end
        self.single = single
        self.empty = empty
        self.is_pointer = is_pointer
        self.simple_mode = simple_mode
        self.no_empty_list = [e for e in [self.begin, self.mid, self.end, self.single] if e]
        self.start_set = set([e for e in [begin, single] if e])
        self.end_set = set([e for e in [end, single] if e])
        self.mid_set = set([mid])

    def get_single(self):
        return self.single if self.single else self.begin

    def get_end(self):
        return self.end if self.end else self.mid

# 将label_type和label_part组合成一个token_label。比如"B"+"人物",编码得到"B_人物"
def encode_label(label_part: str, label_type: str, seq_label_strategy: SeqLabelStrategy) -> str:
    if seq_label_strategy.simple_mode and label_part not in seq_label_strategy.start_set:
        return label_part
    else:
        return LABEL_SEP.join([label_part, label_type])


# 用和encode_label相同的方法，将token的label解码得到将label_type和label_part。比如"B_人物"解码得到"B"+"人物"
def decode_label(label: str) -> Tuple[str, str]:
    fields = label.split(LABEL_SEP)
    label_part = fields[0]
    label_type = LABEL_SEP.join(fields[1:]) if len(fields) > 1 else None
    return label_part, label_type


# 从文件读取所有label，并按照seq_label_strategy编码
def read_seq_label_file(file_path, seq_label_strategy: SeqLabelStrategy):
    label_list = load_lines(file_path)
    return apply_seq_label_strategy(label_list, seq_label_strategy)


def apply_seq_label_strategy(label_list: List[str], seq_label_strategy: SeqLabelStrategy) -> List[str]:
    full_label_list = [] if seq_label_strategy.is_pointer else [seq_label_strategy.empty]
    for label in label_list:
        for label_part in seq_label_strategy.no_empty_list:
            full_label = encode_label(label_part, label, seq_label_strategy)
            if full_label not in full_label_list:
                full_label_list.append(full_label)
    return full_label_list


# 判断一个label在span_extract_strategy编码方式下是不是对应start_label的mid_label
def is_mid(start_label_type, label_part, label_type,
           span_extract_strategy: SeqLabelStrategy):
    if not span_extract_strategy.simple_mode:
        if start_label_type != label_type:
            return False
    if label_part == span_extract_strategy.mid:
        return True
    return False


# 判断一个label在span_extract_strategy编码方式下是不是对应start_label的明确end_label
def is_exact_end(start_label_type, label_part, label_type,
                 span_extract_strategy: SeqLabelStrategy):
    if not span_extract_strategy.simple_mode:
        if start_label_type != label_type:
            return False
    return label_part == span_extract_strategy.end


# 判断一个label在span_extract_strategy编码方式下是不是对应start_label的可能的end_label
def is_valid_end(label_part, gap, span_extract_strategy: SeqLabelStrategy):
    if gap == 0 and label_part == span_extract_strategy.get_single():
        return True
    if gap > 0 and label_part == span_extract_strategy.get_end():
        return True
    return False


# 给定一个不重叠的labeled token序列，以及span的start位置、label。找到符合span_extract_strategy规范的最长span的范围
def get_valid_span(label_list: List[str], start_idx, start_label_type,
                   span_extract_strategy: SeqLabelStrategy):
    pre_idx = start_idx
    for idx in range(start_idx + 1, len(label_list)):
        label_part, label_type = decode_label(label_list[idx])
        if is_exact_end(start_label_type, label_part, label_type, span_extract_strategy):
            return start_idx, idx + 1
        if is_mid(start_label_type, label_part, label_type, span_extract_strategy):
            continue
        pre_idx = idx - 1
        break
    pre_part, _ = decode_label(label_list[pre_idx])
    if is_valid_end(pre_part, pre_idx - start_idx, span_extract_strategy):
        return start_idx, pre_idx + 1
    return None


# 给定一个重叠的labeled token序列，以及span的start位置、label。找到符合span_extract_strategy规范的最长span的范围
def get_overlap_valid_span(label_list: List[Set[str]], start_idx, start_label_type,
                           span_extract_strategy: SeqLabelStrategy):
    if span_extract_strategy.is_pointer:
        for idx in range(start_idx, len(label_list)):
            label_set = label_list[idx]
            decoded_label_list = [decode_label(e) for e in label_set]
            exact_end_list = [e for e in decoded_label_list if
                              is_exact_end(start_label_type, e[0], e[1], span_extract_strategy)]
            if exact_end_list:
                return start_idx, idx + 1
        return None

    idx = start_idx + 1
    while idx < len(label_list):
        label_set = label_list[idx]
        decoded_label_list = [decode_label(e) for e in label_set]
        exact_end_list = [e for e in decoded_label_list if
                          is_exact_end(start_label_type, e[0], e[1], span_extract_strategy)]
        if exact_end_list:
            return start_idx, idx + 1
        mid_list = [e for e in decoded_label_list if
                    is_mid(start_label_type, e[0], e[1], span_extract_strategy)]
        if mid_list:
            idx += 1
            continue
        break
    pre_idx = idx - 1
    pre_label_set = label_list[pre_idx]
    pre_decoded_label_list = [decode_label(e) for e in pre_label_set]
    valid_end_list = [e for e in pre_decoded_label_list if
                      is_valid_end(e[0], pre_idx - start_idx, span_extract_strategy)]
    if valid_end_list:
        return start_idx, pre_idx + 1
    return None


# 给定一个不重叠的labeled token list, 根据span_extract_strategy 解码出所有token级别的span范围和类别
def get_valid_spans(label_list: List[str], span_extract_strategy: SeqLabelStrategy):
    rs_list = []
    for idx, label in enumerate(label_list):
        label_part, label_type = decode_label(label)
        if label_part == span_extract_strategy.single:
            rs_list.append((label_type, (idx, idx + 1)))
        elif label_part == span_extract_strategy.begin:
            valid_span = get_valid_span(label_list, idx, label_type, span_extract_strategy)
            if valid_span:
                rs_list.append((label_type, valid_span))
    rs_list = sorted(rs_list, key=lambda x: x[1])
    return rs_list


# 给定一个重叠的labeled token list, 根据span_extract_strategy 解码出所有token级别的span范围和类别
def get_overlap_valid_spans(label_list: List[Set[str]], span_extract_strategy: SeqLabelStrategy):
    rs_list = []
    for idx, label_set in enumerate(label_list):
        for label in label_set:
            label_part, label_type = decode_label(label)
            if label_part == span_extract_strategy.single:
                rs_list.append((label_type, (idx, idx + 1)))
            elif label_part == span_extract_strategy.begin:
                valid_span = get_overlap_valid_span(label_list, idx, label_type, span_extract_strategy)
                if valid_span:
                    rs_list.append((label_type, valid_span))
    rs_list = sorted(rs_list, key=lambda x: x[1])
    return rs_list


# 根据原始数据以及span_list解码出SeqSpan结构的数据
def decode_text_spans(feature: Dict, spans: List[Tuple[str, Tuple[int, int]]],
                      label_list: List[Dict[str, float]]) -> TextSpans:
    target_list = set()
    token2char = feature["token2char"]
    for label_type, (start, end) in spans:
        end = end - 1
        prob = label_list[start][label_type]
        if token2char[start] and token2char[end]:
            char_start, _ = token2char[start]
            _, char_end = token2char[end]
            text = feature["full_text"][char_start: char_end]
            if text:
                target_list.add(TextSpan(text=text, label=label_type,
                                         span=(char_start, char_end), prob=prob))
    return list(sorted(target_list, key=lambda x: x.span))


# 给定原始数据以及不重叠的token级别label list，解码出SeqSpan的list
def decode_label_sequence(feature: Dict, labels: List[Tuple[str, float]],
                          span_extract_strategy: SeqLabelStrategy) -> TextSpans:
    token_len = len(feature['tokens'])
    labels = labels[:token_len]
    label_values = [e[0] for e in labels]
    label_dicts = [{decode_label(k)[1]: v} for k, v in labels]
    valid_spans = get_valid_spans(label_values, span_extract_strategy)
    targets = decode_text_spans(feature, valid_spans, label_dicts)
    return targets


# 给定原始数据以及重叠的token级别label list，解码出SeqSpan的list
def decode_overlap_label_sequence(feature: Dict, label_list: List[Set[Tuple[str, float]]],
                                  seq_label_strategy: SeqLabelStrategy) -> TextSpans:
    token_len = len(feature["tokens"])
    label_list = label_list[:token_len]
    label_value_list = [{e[0] for e in s} for s in label_list]
    label_dict_list = [{decode_label(k)[1]: v for k, v in item} for item in label_list]

    valid_span_list = get_overlap_valid_spans(label_value_list, seq_label_strategy)

    target_list = decode_text_spans(feature, valid_span_list, label_dict_list)
    return target_list


# 给定token list以及不重叠的char级别的span信息，得到token级别的标注结果
def get_token_label_sequence(token_list, target_list: TextSpans, char2token, seq_label_strategy: SeqLabelStrategy):
    if seq_label_strategy.is_pointer:
        raise Exception(f"pointer strategy only work with multi_label sequence labeling task")
    tokens_label_sequence = [seq_label_strategy.empty for _ in range(len(token_list))]

    def is_all_empty(s, e):
        for i in range(s, e + 1):
            if tokens_label_sequence[i] != seq_label_strategy.empty:
                return False
        return True

    for text_span in target_list:
        text, label, (start, end) = text_span.text, text_span.label, text_span.span
        end = end - 1
        token_start, token_end = char2token[start], char2token[end]
        if is_all_empty(token_start, token_end):
            if token_end == token_start:
                tokens_label_sequence[token_start] = encode_label(seq_label_strategy.get_single(),
                                                                  label, seq_label_strategy)
            else:
                tokens_label_sequence[token_start] = encode_label(seq_label_strategy.begin,
                                                                  label, seq_label_strategy)
                for idx in range(token_start + 1, token_end):
                    tokens_label_sequence[idx] = encode_label(seq_label_strategy.mid, label,
                                                              seq_label_strategy)
                tokens_label_sequence[token_end] = encode_label(seq_label_strategy.get_end(),
                                                                label, seq_label_strategy)
    return tokens_label_sequence


# 给定token list以及重叠的char级别的span信息，得到token级别的标注结果
def get_overlap_token_label_sequence(token_list, target_list: TextSpans, char2token, seq_label_strategy: SeqLabelStrategy):
    tokens_label_sequence = [set() for _ in range(len(token_list))]
    for text_span in target_list:
        text, label, (start, end) = text_span.text, text_span.label, text_span.span
        end = end - 1
        token_start: int = char2token[start]
        token_end: int = char2token[end]
        if seq_label_strategy.is_pointer:
            tokens_label_sequence[token_start].add(
                encode_label(seq_label_strategy.begin, label, seq_label_strategy))
            tokens_label_sequence[token_end].add(
                encode_label(seq_label_strategy.get_end(), label, seq_label_strategy))
        else:
            if token_end == token_start:
                tokens_label_sequence[token_start].add(
                    encode_label(seq_label_strategy.get_single(), label, seq_label_strategy))
            else:
                tokens_label_sequence[token_start].add(
                    encode_label(seq_label_strategy.begin, label, seq_label_strategy))
                for idx in range(token_start + 1, token_end):
                    tokens_label_sequence[idx].add(
                        encode_label(seq_label_strategy.mid, label, seq_label_strategy))
                tokens_label_sequence[token_end].add(
                    encode_label(seq_label_strategy.get_end(), label, seq_label_strategy))
    if not seq_label_strategy.is_pointer:
        for label_set in tokens_label_sequence:
            if not label_set:
                label_set.add(seq_label_strategy.empty)

    return tokens_label_sequence


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


def get_unique_text_span(text_span: TextSpan):
    return (text_span.text, text_span.label, text_span.span)

# 将sequence labeling结果输出
def get_text_span_classify_output(examples: List[UnionTextSpanClassifyExample], preds: List[TextSpans]):
    output = []
    for example, pred in zip(examples, preds):
        rs_item = example.dict()
        rs_item.update(predict=pred)
        if isinstance(example, LabeledTextSpanClassifyExample):
            true_set = set([get_unique_text_span(s) for s in example.label])
            pred_set = set([get_unique_text_span(s) for s in pred])

            tp_set, fp_set, fn_set = get_tp_fp_fn_set(true_set, pred_set)
            rs_item.update(tp_set=tp_set, fp_set=fp_set, fn_set=fn_set)
        output.append(rs_item)
    return output


# 将token_label编码成分类的vector形式
def token_label2classify_label_input(target_token_label_sequence, multi_label, label2id):
    if multi_label:
        classify_label_input = [multi_label2id_vector(e, label2id) for e in target_token_label_sequence]
    else:
        classify_label_input = [label2id[e] for e in target_token_label_sequence]

    return classify_label_input

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

class AbstractTextSpanClassifyModelAIConfig(AIConfigBaseModel):
    example_cls = TextSpanClassifyExample
    labeled_example_cls = LabeledTextSpanClassifyExample
