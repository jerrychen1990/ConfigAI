# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     core
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
import pickle
from abc import abstractmethod, ABC
from typing import Iterable

from snippets import jdumps, jdump, jload, ensure_dir_path, jload_lines

from config_ai.schema import Task

logger = logging.getLogger(__name__)


class AIConfigBaseModel(ABC, object):
    task: Task = None
    """
    所有model的一个基类，可以用底层可以是一个nn model，可以是一个规则系统，也可以是多个其他model构成的pipeline
    一个model用来解决一类给定输入输出的问题
    """

    def __new__(cls, *args, **kwargs):
        cls.input_cls = cls.task.input_cls
        cls.output_cls = cls.task.output_cls
        return super().__new__(cls)

    # 读取配置文件，init一个model实体
    def __init__(self, config):
        self._load_config(config)

    def _load_config(self, config):
        logger.info("loading config...")
        self.config = config
        logger.info("init model with config:")
        logger.info(jdumps(self.config))

        self.config["model_cls"] = self.__class__.__name__
        self.model_name = config.get('model_name', "tmp_model")
        self.task_config = config.get('task_config')

    @classmethod
    def load_examples(cls, data_path: str, return_generator=False) -> Iterable:
        assert cls.task is not None

        if return_generator:
            return (cls.task.input_cls(**e) for e in jload_lines(data_path, return_generator=True))
        else:
            return [cls.task.input_cls(**e) for e in jload_lines(data_path, return_generator=True)]

    @staticmethod
    def get_config_path(path):
        return os.path.join(path, "config.json")

    @staticmethod
    def get_pkl_path(path):
        return os.path.join(path, "model.pkl")

    # save model到path路径下，方便下次可以从path路径复原出model
    @ensure_dir_path
    def save(self, path, save_type="json"):
        logger.info(f"saving model to {path}")
        if save_type == "json":
            jdump(self.config, AIConfigBaseModel.get_config_path(path))
        elif save_type == "pkl":
            pickle.dump(self, open(AIConfigBaseModel.get_pkl_path(path), "wb"))
        else:
            raise ValueError(f"invalid save type:{save_type}")

    # 从$path路径下load出模型
    @classmethod
    def load(cls, path, load_type="json"):
        logger.info(f"loading model from path:{path}")
        if load_type == "json":
            config = jload(AIConfigBaseModel.get_config_path(path))
            model = cls(config=config)
        elif load_type == "pkl":
            model = pickle.load(open(AIConfigBaseModel.get_pkl_path(path), "rb"))
        else:
            raise Exception(f"invalid load type:{load_type}")
        return model

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, **kwargs):
        raise NotImplementedError
