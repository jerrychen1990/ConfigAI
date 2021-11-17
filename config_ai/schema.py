# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     schema
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
from typing import Union, List
from pydantic import BaseModel, Field


class Label(BaseModel):
    name: str = Field(description="标签名称")
    prob: float = Field(description="标签概率", le=1., ge=0, default=1.)


LabelOrLabels = Union[List[Label], Label]


class TextClassifyExample(BaseModel):
    text: str = Field(description="待分类的文本")


class LabeledTextClassifyExample(TextClassifyExample):
    label: LabelOrLabels = Field(description="标签或者标签列表(多分类)")


UnionTextClassifyExample = Union[LabeledTextClassifyExample, TextClassifyExample]

#
# # 关系分类数据
# class RelationClassifyExample(TextClassifyExample):
#     text_span1: TextSpan = Field(description="第一个text span")
#     text_span2: TextSpan = Field(description="第二个text span")
#
#
# # 关系分类带标签数据
# class LabeledRelationClassifyExample(RelationClassifyExample):
#     label: LabelOrLabels = Field(description="关系标签，可以是单标签也可以是多标签")
#
#
# UnionRelationClassifyExample = Union[RelationClassifyExample, LabeledRelationClassifyExample]
#
# MaskedToken = Tuple[int, str]
#
# # 带有标注的MLM任务数据
# class MaskedLanguageModelExample(TextClassifyExample):
#     masked_tokens: Optional[List[str]] = Field(description="被masked掉的token值")
