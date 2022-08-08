# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     __init__.py
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
import logging
import os

from snippets.utils import jload
from config_ai.models.core import AIConfigBaseModel
from config_ai.models.text_classify import CLSTokenClassifyModel
# from config_ai.models.mlm import TransformerMLMModel
# from config_ai.models.relation_classify import RelationTokenClassifyModel
# from config_ai.models.text_classify import CLSTokenClassifyModel, MLMTextClassifyModel
# from config_ai.models.text_span_classify import SeqLabelingModel, GlobalPointerModel
# from config_ai.models.seq2seq import TransformerSeq2SeqModel

logger = logging.getLogger(__name__)

# ALL_MODELS = [CLSTokenClassifyModel, MLMTextClassifyModel] + \
#              [SeqLabelingModel, GlobalPointerModel] + \
#              [RelationTokenClassifyModel] + \
#              [TransformerMLMModel] + \
#              [TransformerSeq2SeqModel]

ALL_MODELS = [CLSTokenClassifyModel]

_ALL_MODEL_DICT = {cls.__name__: cls for cls in ALL_MODELS}


def get_model_class_by_name(model_class_name):
    model_cls = _ALL_MODEL_DICT.get(model_class_name)
    if not model_cls:
        raise ValueError(
            f"not valid model_cls:{model_class_name}, "
            f"valid model_cls list:{sorted(_ALL_MODEL_DICT.keys())}"
        )
    return model_cls


def load_model(path, load_type="json", load_nn_model=True) -> AIConfigBaseModel:
    config_path = os.path.join(path, "config.json")
    config = jload(config_path)
    model_cls = config["model_cls"]
    model_cls = get_model_class_by_name(model_cls)
    return model_cls.load(path=path, load_type=load_type, load_nn_model=load_nn_model)
