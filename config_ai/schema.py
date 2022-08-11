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
from enum import Enum
from typing import Union, List, Tuple, Optional
from pydantic import BaseModel, Field


# 基础数据类型
class HashableModel(BaseModel):
    class Config:
        frozen = True
        allow_population_by_field_name = True


# 模型输入
class Example(HashableModel):
    pass


# 模型输出
class Predict(BaseModel):
    pass


# 文本模型输入
class TextExample(BaseModel):
    text: str = Field(description="输入的文本")


# 标签
class Label(HashableModel):
    name: str = Field(description="标签名称")
    prob: float = Field(description="标签概率", le=1., ge=0, default=1.)


Labels = List[Label]
# 分类模型输出
LabelOrLabels = Union[Labels, Label]


#

# 文本模型输入
class TextClassifyExample(TextExample):
    label: Optional[LabelOrLabels] = Field(description="Ground Truth, 训练数据有此字段")


class Task(Enum):
    TEXT_CLS = ("text-classification", TextClassifyExample, LabelOrLabels)

    def __init__(self, hf_task, input_cls, output_cls):
        self.hf_task = hf_task
        self.input_cls = input_cls
        self.output_cls = output_cls

#
#
#
# 文本片段
# class TextSpan(HashableModel):
#     text: str = Field(description="文本片段内容")
#     span: Tuple[int, int] = Field(description="文本片段的下标区间，前闭后开")
#     label: str = Field(default="片段标签")
#     prob: float = Field(description="文本片段分类的概率值", default=1., ge=0., le=1.)
#
#
# TextSpans = List[TextSpan]
#
#
# # 文本片段分类数据
# class TextSpanClassifyExample(TextExample):
#     pass
#
#
# class LabeledTextSpanClassifyExample(TextSpanClassifyExample):
#     text_spans: TextSpans = Field(description="文本片段列表")
#
#
# UnionTextSpanClassifyExample = Union[LabeledTextSpanClassifyExample, TextSpanClassifyExample]


#
# # 关系分类数据
# class RelationClassifyExample(TextExample):
#     text_span1: TextSpan = Field(description="第一个text span")
#     text_span2: TextSpan = Field(description="第二个text span")
#
#
# # 关系分类带标签数据
# class LabeledRelationClassifyExample(RelationClassifyExample):
#     label: LabelOrLabels = Field(description="关系标签，可以是单标签也可以是多标签")
#
#
# UnionRelationClassifyExample = Union[LabeledRelationClassifyExample, RelationClassifyExample]
#
#
# # MLM任务数据
# class MLMExample(BaseModel):
#     text: str = Field(description="原始文本")
#     masked_tokens: Optional[List[str]] = Field(description="被masked掉的token值")
#
#
# MASK = '[MASK]'
#
#
# # Seq2Seq任务数据
# class Seq2SeqExample(TextExample):
#     pass
#
#
# class GenText(HashableModel):
#     text: str = Field(description="生成的文本")
#     prob: float = Field(description="生成文本的probability", ge=0., le=1., default=1.)

#
# class LabeledSeq2SeqExample(Seq2SeqExample):
#     tgt_text: GenText = Field(description="目标文本")
#
# #
# UnionSeq2SeqExample = Union[LabeledSeq2SeqExample, Seq2SeqExample]
