# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tf_seq_labeling
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
import logging
from enum import unique, Enum
from typing import Dict, List, Tuple, Set

from snippets import seq2dict, load_lines
from transformers import AutoModelForTokenClassification

from config_ai.models.huggingface_core import HuggingfaceBaseModel
from config_ai.models.text_span_classify.common import AbstractTextSpanClassifyModelAIConfig
from config_ai.schema import Example, TextSpans, TextSpan, TextSpanClassifyExample, Label

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


def get_char2token(text, offset_map):
    char2token = [-1] * len(text)
    for idx, (s, e) in enumerate(offset_map):
        for i in range(s, e):
            char2token[i] = idx
    return char2token


# 给定token list以及不重叠的char级别的span信息，得到token级别的标注结果
def get_token_label_sequence(token_len, text_spans: TextSpans, char2token, seq_label_strategy: SeqLabelStrategy):
    if seq_label_strategy.is_pointer:
        raise Exception(f"pointer strategy only work with multi_label sequence labeling task")
    tokens_label_sequence = [seq_label_strategy.empty for _ in range(token_len)]

    def is_all_empty(s, e):
        for i in range(s, e + 1):
            if tokens_label_sequence[i] != seq_label_strategy.empty:
                return False
        return True

    for text_span in text_spans:
        text, label, (start, end) = text_span.text, text_span.label.name, text_span.span
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


def decode_hf_entities(pred: List[dict], seq_label_strategy: SeqLabelStrategy) -> TextSpans:
    text_spans = []
    cur_type = None
    acc = []

    def gather():
        if not cur_type or not acc:
            return None, []
        text = "".join([e["word"] for e in acc])
        span = [acc[0]["start"], acc[-1]["end"]]
        # todo 是否有更合理的方案？
        score = min(e["score"] for e in acc)

        text_spans.append(TextSpan(text=text, span=span, label=Label(name=cur_type, score=score)))
        return None, []

    for item in pred:
        label, score, word, start, end = item["entity"], item["score"], item["word"], item["start"], item["end"]
        label_part, label_type = decode_label(label)

        if cur_type and label_type != cur_type:
            cur_type, acc = gather()

        if label_part in [seq_label_strategy.single, seq_label_strategy.begin]:
            cur_type = label_type
            acc = [item]

        if label_part in [seq_label_strategy.single, seq_label_strategy.end]:
            cur_type, acc = gather()

        if label_part == seq_label_strategy.mid and label_type == cur_type:
            acc.append(item)

    gather()

    return text_spans


class SeqLabelingModel(AbstractTextSpanClassifyModelAIConfig, HuggingfaceBaseModel):
    auto_model_cls = AutoModelForTokenClassification

    def _load_config(self, config):
        super()._load_config(config)
        self.seq_label_strategy: SeqLabelStrategy = SeqLabelStrategy[
            self.task_config['seq_label_strategy']]

        self.max_len = self.task_config['max_len']
        self.multi_label = self.task_config.get("multi_label", False)
        self.label_list = read_seq_label_file(self.task_config['label_file_path'], self.seq_label_strategy)
        self.label2id, self.id2label = seq2dict(self.label_list)
        self.label_num = len(self.label2id)

    def build_model(self, **kwargs):
        self.nn_model = self.auto_model_cls.from_pretrained(self.pretrained_model_path, id2label=self.id2label)
        return self.nn_model

    def _train_preprocess(self, examples: List[Dict], truncation=True):
        rs = self.tokenizer(examples["text"], truncation=True, return_offsets_mapping=True, max_length=self.max_len)
        char2token = [get_char2token(text, offset_map) for text, offset_map in
                      zip(examples["text"], rs["offset_mapping"])]

        tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in rs["input_ids"]]

        ner_tag_labels = [get_token_label_sequence(len(token_ids),
                                                   [TextSpan(**e) for e in text_spans], c2t,
                                                   self.seq_label_strategy)
                          for (token_ids,
                               text_spans,
                               c2t) in zip(rs["input_ids"], examples["text_spans"], char2token)]
        ner_tag = [[self.label2id[l] for l in tag_label] for tag_label in ner_tag_labels]

        rs.update(labels=ner_tag, tokens=tokens)
        return rs

    def _predict_preprocess(self, example: TextSpanClassifyExample):
        return example.text

    def _predict_postprocess(self, pred) -> TextSpans:
        text_spans = decode_hf_entities(pred, self.seq_label_strategy)
        return text_spans

        # labels = []
        # for i, label in enumerate(examples[f"ner_tags"]):
        #     word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        #     previous_word_idx = None
        #     label_ids = []
        #     for word_idx in word_ids:  # Set the special tokens to -100.
        #         if word_idx is None:
        #             label_ids.append(-100)
        #         elif word_idx != previous_word_idx:  # Only label the first token of a given word.
        #             label_ids.append(label[word_idx])
        #         else:
        #             label_ids.append(-100)
        #         previous_word_idx = word_idx
        #     labels.append(label_ids)
        #
        # tokenized_inputs["labels"] = labels
        # return tokenized_inputs
        #
        # pass

    #
    # def compile_model(self, optimizer_name: str, optimizer_args: dict, **kwargs):
    #     logger.info(f"compile model with optimizer_name:{optimizer_name}, optimizer_args:{optimizer_args}")
    #     with self.get_scope():
    #         classify_labels = Input(shape=(None, self.label_num) if self.multi_label else (None,),
    #                                 name='classify_labels', dtype=tf.int32)
    #         token_ids, segment_ids = self.nn_model.inputs
    #         output = self.nn_model([token_ids, segment_ids])
    #         self.train_model = Model(inputs=[token_ids, segment_ids, classify_labels], outputs=[output])
    #
    #     loss_mask = Lambda(function=lambda x: tf.cast(tf.not_equal(x, 0), tf.float32), name="pred_mask")(token_ids)
    #
    #     # 计算loss的时候，过滤掉pad token的loss
    #     loss_layer = build_classify_loss_layer(multi_label=self.multi_label, with_mask=True)
    #     loss = loss_layer([classify_labels, output, loss_mask])
    #     self.train_model.add_loss(loss)
    #
    #     # 计算accuracy的时候，过滤掉pad token 的accuracy
    #     masked_accuracy_func = masked_binary_accuracy if self.multi_label else masked_sparse_categorical_accuracy
    #     metric_layer = MetricLayer(masked_accuracy_func)
    #     masked_accuracy = metric_layer([classify_labels, output, loss_mask])
    #     self.train_model.add_metric(masked_accuracy, aggregation="mean", name="accuracy")
    #
    #     optimizer = OptimizerFactory.create(optimizer_name, optimizer_args)
    #     self.train_model.compile(optimizer=optimizer)
    #     logger.info("training model's summary:")
    #     self.train_model.summary(print_fn=logger.info)
    #     self._update_model_dict("train", self.train_model)
    #
    # def example2feature(self, example: UnionTextSpanClassifyExample) -> Dict:
    #     feature = self.tokenizer.do_tokenize(text=example.text, store_map=True)
    #     if isinstance(example, LabeledTextSpanClassifyExample):
    #         feature.update(text_spans=[e.dict(exclude_none=True) for e in example.text_spans])
    #     return feature
    #
    # def _feature2records(self, idx, feature: Dict, mode: str) -> List[Dict]:
    #     record = dict(idx=idx, **feature)
    #     if mode == "train":
    #         text_spans = feature.get("text_spans")
    #         if text_spans is None:
    #             raise ValueError(f"not text_spans key found in train mode!")
    #         text_spans = [TextSpan(**e) for e in text_spans]
    #         token_label_func = get_overlap_token_label_sequence if self.multi_label else get_token_label_sequence
    #         target_token_label_sequence = token_label_func(feature["tokens"], text_spans,
    #                                                        feature["char2token"], self.seq_label_strategy)
    #         classify_labels = token_label2classify_label_input(target_token_label_sequence, self.multi_label,
    #                                                            self.label2id)
    #         record.update(target_token_label_sequence=target_token_label_sequence, classify_labels=classify_labels)
    #
    #     truncate_record(record=record, max_len=self.max_len,
    #                     keys=["token_ids", "segment_ids", "tokens", "classify_labels"])
    #
    #     return [record]
    #
    # @discard_kwarg
    # @log_cost_time
    # def _post_predict(self, features, pred_tensors, show_detail, threshold=0.5) -> List[TextSpans]:
    #     def _tensor2output(feature, pred_tensor) -> TextSpans:
    #         pred_labels = tensor2labels(pred_tensor, self.multi_label, self.id2label, threshold=threshold)
    #         tokens = feature["tokens"]
    #         pred_labels = pred_labels[:len(tokens)]
    #         if show_detail:
    #             logger.info(f"tokens:{tokens}")
    #             for idx, (token, pred_label) in enumerate(zip(tokens, pred_labels)):
    #                 if pred_label and pred_label != self.seq_label_strategy.empty:
    #                     logger.info(f"idx:{idx}, token:{token}, pred:{pred_label}")
    #         decode_func = decode_overlap_label_sequence if self.multi_label else decode_label_sequence
    #         text_spans = decode_func(feature, pred_labels, self.seq_label_strategy)
    #         return text_spans
    #
    #     preds = [_tensor2output(f, p) for f, p in zip(features, pred_tensors)]
    #     return preds
