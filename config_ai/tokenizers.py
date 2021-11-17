# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tokenizers
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
import hashlib
import logging
import bert4keras

from abc import ABC, abstractmethod
from typing import List, Union

from bert4keras.tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer
from snippets.utils import load_lines

logger = logging.getLogger(__name__)


def replace_unused_tokens(vocabs: List[str], tokens: List[str], start_idx=10, end_idx=100):
    assert len(tokens) <= end_idx - start_idx
    for idx, token in enumerate(tokens):
        vocabs[start_idx + idx] = token

    return vocabs


def get_token2char_index_mapping(tokened, sep_idx: int) -> dict:
    """
    通过tokenizer.encode的到的结果$tokened，sep_idx。获得token到对应字符下标的dict
    Args:
        tokened:通过tokenizer.encode的到的结果
        sep_idx:bert的第二句在模型输入中的字符下标

    Returns:

    """
    token2char = []
    tokens = tokened.tokens
    offsets = tokened.offsets

    sep_token_num = 0
    for token, offset in zip(tokens, offsets):
        if sep_token_num >= 2 and sep_idx:
            offset = (offset[0] + sep_idx, offset[1] + sep_idx)
        if offset[0] == offset[1]:
            sep_token_num += 1
        token2char.append(offset)
    assert len(token2char) == len(tokens)
    return token2char


# 通过token2char的映射，反过来计算char2token的映射关系
def get_char2token_mapping(sentence: str, token2char: dict) -> dict:
    char2token = [-1] * len(sentence)
    for idx, span in enumerate(token2char):
        if span:
            start, end = span
            char2token[start:end] = [idx] * (end - start)
    return char2token


def read_vocabs(vocabs: Union[str, List]):
    if isinstance(vocabs, str):
        vocabs = load_lines(vocabs)
    token2id = {token: idx for idx, token in enumerate(vocabs)}
    id2token = {idx: token for idx, token in enumerate(vocabs)}
    return vocabs, token2id, id2token


class AbstractTokenizer(ABC):
    _MASK = "[MASK]"

    def __init__(self, vocabs, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.vocabs, self._token2id, self._id2token = read_vocabs(vocabs)
        self.special_tokens = [e for e in self.vocabs if e.startswith("[") and e.endswith("]") and "unused" not in e]

    def token2id(self, token: str) -> int:
        return self._token2id[token]

    def id2token(self, token_id: int) -> str:
        return self._id2token[token_id]

    @property
    @abstractmethod
    def end_token(self):
        pass

    @property
    def end_token_id(self):
        return self.token2id(self.end_token)

    @property
    @abstractmethod
    def start_token(self):
        pass

    @property
    def invisible_tokens(self):
        return [self.start_token, self.end_token]


    @property
    def start_token_id(self):
        return self.token2id(self.start_token)

    @property
    def mask_token_id(self):
        return self.token2id(self._MASK)

    def do_tokenize(self, text, extra_text: str = None, store_map=False) -> dict:
        tokened = self._tokenizer.encode(text, extra_text)
        full_text = text + extra_text if extra_text else text
        sep_idx = len(text) if extra_text else None

        detail_result = dict(full_text=full_text, text=text, extra_text=extra_text,
                             token_ids=tokened.ids, segment_ids=tokened.type_ids, tokens=tokened.tokens)
        if store_map:
            token2char = get_token2char_index_mapping(tokened, sep_idx)
            char2token = get_char2token_mapping(text, token2char)
            detail_result.update(token2char=token2char, char2token=char2token)
        return detail_result

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def get_cache_tag(self):
        return self.__class__.__name__


class HFWordPieceTokenizer(AbstractTokenizer):
    def __init__(self, vocabs: List[str], do_lower_case=True, **kwargs):
        super().__init__(vocabs=vocabs, do_lower_case=do_lower_case)
        self._tokenizer: BertWordPieceTokenizer = BertWordPieceTokenizer(vocab=self._token2id,
                                                                         lowercase=self.do_lower_case, **kwargs)
        self.vocabs = [e[0] for e in sorted(self._tokenizer.get_vocab().items(), key=lambda x: x[1])]
        self.vocab_size = self._tokenizer.get_vocab_size()
        self._tokenizer.add_special_tokens(self.special_tokens)

    def get_cache_tag(self):
        lower_tag = "l" if self.do_lower_case else "u"
        vocabs = "".join(self.vocabs)
        md5 = hashlib.md5(vocabs.encode("utf8"))
        return f"{self.__class__.__name__}_{lower_tag}_{md5.hexdigest()}"

    @property
    def end_token(self):
        return self._tokenizer._parameters["sep_token"]

    @property
    def start_token(self):
        return self._tokenizer._parameters["cls_token"]


class Bert4kerasTokenizer(AbstractTokenizer):
    _replace_chars = "厷厸厹厺厼厽厾叀叁参叄叅叆叇亝収叏叐叒叓叕叚叜叝叞叠壭壱売壳壴壵壶壷壸壶壻壸壾壿夀夁"

    def __init__(self, vocabs, **kwargs):
        super().__init__(vocabs=vocabs)
        if len(self.special_tokens) > len(self._replace_chars):
            raise Exception(
                f"{len(self._replace_chars)} replace_chars not enough for {len(self.special_tokens)} special tokens")
        self.vocab_size = len(self.vocabs)

        self.token_map = dict(zip(self._replace_chars, self.special_tokens))
        logger.info(f"token_map:{self.token_map}")
        self._tokenizer: bert4keras.tokenizers.Tokenizer \
            = bert4keras.tokenizers.Tokenizer(token_dict=self._token2id,
                                              do_lower_case=self.do_lower_case,
                                              word_maxlen=512,
                                              token_translate=self.token_map
                                              )

    @property
    def end_token(self):
        return self._tokenizer._token_end

    @property
    def start_token(self):
        return self._tokenizer._token_start


TOKENIZER_MAP = {
    "bert_word_piece": HFWordPieceTokenizer,
    "bert4keras": Bert4kerasTokenizer
}


def build_tokenizer(cls, args) -> AbstractTokenizer:
    if isinstance(cls, str):
        cls = TOKENIZER_MAP[cls]
    tokenizer = cls(**args)
    return tokenizer
