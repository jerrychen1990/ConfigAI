# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     common
   Description :
   Author :       chenhao
   date：          2021/4/14
-------------------------------------------------
   Change Activity:
                   2021/4/14:
-------------------------------------------------
"""
from abc import ABC

from config_ai.evaluate import eval_text_classify
from config_ai.models import AIConfigBaseModel
from config_ai.models.text_classify.common import get_text_classify_output
from config_ai.schema import LabeledRelationClassifyExample
# 测评文本分类的结果
from config_ai.schema import RelationClassifyExample

get_relation_classify_output = get_text_classify_output


class AbstractRelationClassifyModel(AIConfigBaseModel, ABC):
    example_cls = RelationClassifyExample
    labeled_example_cls = LabeledRelationClassifyExample
