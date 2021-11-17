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
from typing import Dict, Iterable, List

from snippets import jdumps, jdump, jload, jload_lines, load_lines, ensure_dir_path
from tensorflow.python.keras.models import Model

from config_ai.tokenizers import build_tokenizer
from snippets import PythonObjectEncoder

logger = logging.getLogger(__name__)

class ModelEncoder(PythonObjectEncoder):
    def default(self, obj):
        if isinstance(obj, Model):
            return obj.name
        return super().default()

class AIConfigBaseModel(ABC, object):
    example_cls = None
    labeled_example_cls = None

    """
    所有model的一个基类，可以用底层可以是一个nn model，可以是一个规则系统，也可以是多个其他model构成的pipeline
    一个model用来解决一类给定输入输出的问题
    """

    # 读取配置文件，init一个model实体
    def __init__(self, config):
        self._load_config(config)

    def _load_config(self, config):
        logger.info("init model with config:")
        self.config = config
        self.config["model_cls"] = self.__class__.__name__
        # logger.info(jdumps(self.config))
        self.model_name = config.get('model_name', "tmp_model")
        self.task_config = config.get('task_config')

    @classmethod
    def jload_lines(cls, data_path, max_data_num=None, return_generator=False) -> Iterable:
        def _load_item(d: dict):
            try:
                item = cls.labeled_example_cls(**d)
                return item
            except:
                return cls.example_cls(**d)

        rs = jload_lines(data_path, max_data_num=max_data_num, return_generator=return_generator)
        if return_generator:
            return (_load_item(e) for e in rs)
        return [_load_item(e) for e in rs]

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
            jdump(self.config, AIConfigBaseModel.get_config_path(path), encoder=ModelEncoder)
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

    # 初始化model
    @abstractmethod
    def build_model(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, **kwargs):
        raise NotImplementedError


class NNBasedModelAIConfig(AIConfigBaseModel):
    """
    底层包含Neural Network的model
    """

    def __init__(self, config):
        super().__init__(config)
        self.nn_model = None
        self.model_dict = self.config.get("model_dict", dict())
        self.config["model_dict"] = self.model_dict
        self.tokenizer_config = self.config['tokenizer_config']
        self._init_tokenizer()

    def _init_tokenizer(self):
        logger.info(f"initializing tokenizer with config:\n{jdumps(self.tokenizer_config)}")
        self.tokenizer = build_tokenizer(self.tokenizer_config["tokenizer_name"],
                                         self.tokenizer_config["tokenizer_args"])

        if self.task_config.get("keep_token_path"):
            self.keep_tokens = load_lines(self.task_config["keep_token_path"])
            self.keep_token_ids = [self.tokenizer.token2id(t) for t in self.keep_tokens]
            self.tokenizer_config["tokenizer_args"]["vocabs"] = self.keep_tokens
            logger.info("reinitializing tokenizer with keep_tokens")
            self.tokenizer = build_tokenizer(self.tokenizer_config["tokenizer_name"],
                                             self.tokenizer_config["tokenizer_args"])

        self.vocab_size = self.tokenizer.vocab_size

        logger.info(f"tokenizer initialized with {self.vocab_size} vocabs")

    @ensure_dir_path
    def save(self, path, save_type="json", **kwargs):
        super().save(path=path, save_type=save_type)
        self.save_nn_model(path=path, **kwargs)
        logger.info("save model done")

    def save_nn_model(self, path, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, load_type="json", load_nn_model=True):
        model = super().load(path=path, load_type=load_type)
        if load_nn_model:
            model.load_nn_model(path=path)
        logger.info("load model done")
        return model

    @abstractmethod
    def load_nn_model(self, path):
        raise NotImplementedError

    # compile nn_model（比如设置loss，metrics, optimizer）
    @abstractmethod
    def compile_model(self, **kwargs):
        raise NotImplementedError

    # 原始输入的json数据到增强的json数据（可以做缓存）
    @abstractmethod
    def _example2feature(self, example) -> Dict:
        raise NotImplementedError

    # 增强的json数据到nn_model的输入数据
    @abstractmethod
    def _feature2records(self, idx: int, feature: Dict, mode: str) -> List[Dict]:
        raise NotImplementedError
