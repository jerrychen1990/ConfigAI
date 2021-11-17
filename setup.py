# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     setup
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""

from setuptools import setup, find_packages

REQUIREMENTS = [
    "requests",
    "tensorflow",
    "fire",
    "tqdm",
    "tokenizers",
    "numpy",
    "pydantic",
    "jsonpath-ng",
    "jieba",
    "bert4keras",
    "retrying"
]

setup(
    name='config_ai',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    package_dir={"": "."},
    package_data={},
    url='git@github.com:jerrychen1990/ConfigAI.git',
    license='MIT',
    author='Chen Hao',
    author_email='jerrychen1990@gmail.com',
    zip_safe=True,
    description='write config files for ai model',
    install_requires=REQUIREMENTS
)
