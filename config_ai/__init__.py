# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     __init__.py
   Description :
   Author :       chenhao
   date：          2021/3/29
-------------------------------------------------
   Change Activity:
                   2021/3/29:
-------------------------------------------------
"""
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d]:%(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')

# 使用bert4keras需要设定的全局变量
os.environ['TF_KERAS'] = '1'
