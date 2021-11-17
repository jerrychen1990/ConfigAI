# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     optimizers
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""

from tensorflow.keras.optimizers import (
    Adadelta,
    Adagrad,
    Adamax,
    Nadam,
    RMSprop,
    SGD,
    Adam
)
from bert4keras.optimizers import extend_with_exponential_moving_average, extend_with_piecewise_linear_lr, \
    extend_with_gradient_accumulation

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
# 变成带分段线性学习率的Adam
AdamLR = extend_with_piecewise_linear_lr(Adam, 'AdamLR')
# 梯度累积的Adam
AdamAcc = extend_with_gradient_accumulation(Adam, 'AdamAcc')
# 梯度累积的分段线性学习率Adam
AdamAccLR = extend_with_piecewise_linear_lr(AdamAcc, 'AdamAccLR')


class OptimizerFactory:
    _BUILDERS = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
        "adamema": AdamEMA,
        "adam_lr": AdamLR,
        "adam_acc": AdamAcc,
        "adam_acc_lr": AdamAccLR
    }

    @classmethod
    def create(cls, optimizer_name: str, optimizer_args: dict):
        builder = cls._BUILDERS.get(optimizer_name.lower())
        if not builder:
            raise ValueError(
                f"not valid optimizer_name:{optimizer_name}, "
                f"valid optimizer_name list:{sorted(cls._BUILDERS.keys())}"
            )
        return builder(**optimizer_args)

