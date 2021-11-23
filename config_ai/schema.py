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
from typing import Union, List, Tuple, Optional
from pydantic import BaseModel, Field


# 基础数据类型

class HashableModel(BaseModel):
    class Config:
        frozen = True


# 标签
class Label(HashableModel):
    name: str = Field(description="标签名称")
    prob: float = Field(description="标签概率", le=1., ge=0, default=1.)


Labels = List[Label]
LabelOrLabels = Union[Labels, Label]


# 文本片段
class TextSpan(HashableModel):
    text: str = Field(description="文本片段内容")
    span: Tuple[int, int] = Field(description="文本片段的下标区间，前闭后开")
    label: str = Field(default="片段标签")
    prob: float = Field(description="文本片段分类的概率值", default=1., ge=0., le=1.)


TextSpans = List[TextSpan]


# 模型输入输出

class TextClassifyExample(BaseModel):
    text: str = Field(description="待分类的文本")


class LabeledTextClassifyExample(TextClassifyExample):
    label: LabelOrLabels = Field(description="标签或者标签列表(多分类)")


UnionTextClassifyExample = Union[LabeledTextClassifyExample, TextClassifyExample]


# 文本片段分类数据
class TextSpanClassifyExample(TextClassifyExample):
    pass


class LabeledTextSpanClassifyExample(TextSpanClassifyExample):
    text_spans: TextSpans = Field(description="文本片段列表")


UnionTextSpanClassifyExample = Union[LabeledTextSpanClassifyExample, TextSpanClassifyExample]


# 关系分类数据
class RelationClassifyExample(TextClassifyExample):
    text_span1: TextSpan = Field(description="第一个text span")
    text_span2: TextSpan = Field(description="第二个text span")


# 关系分类带标签数据
class LabeledRelationClassifyExample(RelationClassifyExample):
    label: LabelOrLabels = Field(description="关系标签，可以是单标签也可以是多标签")


UnionRelationClassifyExample = Union[LabeledRelationClassifyExample, RelationClassifyExample]


# MLM任务数据
class MLMExample(BaseModel):
    text: str = Field(description="原始文本")
    masked_tokens: Optional[List[str]] = Field(description="被masked掉的token值")


MASK = '[MASK]'
