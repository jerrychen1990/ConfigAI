# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tf_core
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""

import copy
import json
import logging
import math
import os
import time
import numpy as np
import requests
import tensorflow as tf

from retrying import retry
from tensorflow.python.keras.models import Model, InputLayer
from tqdm import tqdm
from abc import abstractmethod
from typing import List, Dict, Iterable

from config_ai.constants import *
from config_ai.data_utils import DataManager
from config_ai.models.core import NNBasedModelAIConfig
from config_ai.utils import log_cost_time, jdumps, execute_cmd

logger = logging.getLogger(__name__)


def get_keras_model_path(path: str, format) -> str:
    if format == "h5":
        rs_path = os.path.join(path, NN_MODEL_DIR_NAME, H5_FILE_NAME)
    else:
        rs_path = os.path.join(path, NN_MODEL_DIR_NAME)
    return rs_path


def get_tf_serving_model_path(path: str, tf_serving_version: int) -> str:
    return os.path.join(path, NN_MODEL_DIR_NAME, TF_SERVING_DIR_NAME, str(tf_serving_version))


@retry(stop_max_attempt_number=3, wait_random_min=5000, wait_random_max=10000)
def save_keras_model(model: Model, path: str, fmt: str = None, tf_serving_version: int = None) -> None:
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    if fmt:
        keras_model_path = get_keras_model_path(path=path, format=fmt)
        logger.info(f"saving keras model to path:{keras_model_path}")
        dir_path = os.path.dirname(keras_model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.save(keras_model_path, include_optimizer=False, save_format=fmt)
    if tf_serving_version:
        tf_serving_model_path = get_tf_serving_model_path(path=path, tf_serving_version=tf_serving_version)
        dir_path = os.path.dirname(tf_serving_model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        logger.info(f"saving tf serving model to path:{tf_serving_model_path}")
        model.save(tf_serving_model_path, save_format="tf")

        logger.info(f"compress... tf serving model(for euler deployment)...")
        cmd = f"cd {os.path.dirname(tf_serving_model_path)}; tar czvf {tf_serving_version}.tar.gz {tf_serving_version}"
        execute_cmd(cmd)


def infer_with_tf_serving(test_batches: Iterable, tf_serving_url, show_detail=False):
    pred_data = []
    for idx, batch in enumerate(test_batches):
        data = {k: v.tolist() for k, v in batch.items()}
        data = dict(inputs=data)
        logger.info(f"requesting batch{idx}")
        st = time.time()
        if show_detail:
            logger.info(f"request data:\n{json.dumps(data, ensure_ascii=False)}")

        resp = requests.post(url=tf_serving_url, json=data)
        if resp.status_code != 200:
            msg = f"request to {tf_serving_url} failed!\nstatus_code:{resp.status_code}, message:{resp.content}"
            logger.error(msg)
            raise Exception(msg)
        else:
            if show_detail:
                logger.info(f"response data:\n{json.dumps(resp.json()['outputs'], ensure_ascii=False)}")
            cost = time.time() - st
            logger.info(f"request to tf serving cost:{cost:3.3f} seconds")
            pred_data.extend(resp.json()['outputs'])
    pred_data = np.array(pred_data)
    return pred_data


class TFBasedModel(NNBasedModelAIConfig):
    """
    底层nn model使用tf2.1框架的model
    """
    custom_objects = dict()

    def __init__(self, config):
        super().__init__(config)
        self.nn_model: tf.keras.models.Model = None
        self.train_model: tf.keras.models.Model = None
        self.strategy = None

        # 获得当前模型的分布式训练strategy

    def get_strategy(self):
        if not self.strategy:
            self.strategy = tf.distribute.MirroredStrategy()
        return self.strategy

    def get_scope(self):
        strategy = self.get_strategy()
        gpu_num = self.strategy.num_replicas_in_sync
        if gpu_num > 1:
            logger.info(f"number of devices: {gpu_num}, use {self.strategy}'s scope")
            return strategy.scope()
        else:
            logger.info(f"number of devices: {gpu_num}, use SINGLE scope")
            return SingleScope()

    def __getstate__(self):
        odict = copy.copy(self.__dict__)
        for key in ['nn_model', 'tokenizer', 'train_model', 'strategy']:
            if key in odict.keys():
                del odict[key]
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_tokenizer()

    def save_nn_model(self, path, fmt="h5", tf_serving_version=None):
        """
        保存tensorflow模型
        Args:
            path: nn model保存的路径
            fmt: 保存的keras文件格式
            tf_serving_version: tf serving格式的文件的version。None的话表示不保存tf-serving格式文件
        Returns:
        """
        if self.nn_model:
            save_keras_model(model=self.nn_model, path=path, fmt=fmt, tf_serving_version=tf_serving_version)

    # 加载 nn model
    def load_nn_model(self, path):
        for fmt in ["h5", "tf"]:
            keras_model_path = get_keras_model_path(path, format=fmt)
            if os.path.exists(keras_model_path):
                logger.info(f"loading keras model from path:{keras_model_path} with format:{fmt}")
                self.nn_model = tf.keras.models.load_model(keras_model_path, custom_objects=self.custom_objects,
                                                           compile=False)
                self.nn_model.summary(print_fn=logger.info)
                self._update_model_dict("test", self.nn_model)
                return
        raise Exception(f"no valid model file under path:{path}!")

    def train(self, train_data, epochs, batch_size,
              dev_data=None,
              callbacks=[], buffer_size=1024, verbose=1, overwrite_cache=False, **kwargs):
        train_data_manager = DataManager.get_instance(model=self, data=train_data)
        train_data_manager.store_features(overwrite_cache=overwrite_cache)

        dataset = train_data_manager.get_dataset(mode="train")
        data_size = int(dataset.reduce(0, lambda x, _: x + 1))
        logger.info(f"train on {data_size} tensors")
        if "steps_per_epoch" not in kwargs:
            kwargs["steps_per_epoch"] = int(math.ceil(data_size / batch_size))
        if not buffer_size:
            buffer_size = data_size
        train_dataset = train_data_manager.get_train_dataset(repeat=None, batch_size=batch_size,
                                                             buffer_size=min(buffer_size, data_size))

        dev_dataset = None
        if dev_data:
            dev_data_manager = DataManager.get_instance(model=self, data=dev_data)
            dev_data_manager.store_features(overwrite_cache=overwrite_cache)
            dev_dataset = dev_data_manager.get_train_dataset(repeat=1, batch_size=batch_size, buffer_size=None)

        self.train_model.fit(train_dataset, validation_data=dev_dataset, epochs=epochs, callbacks=callbacks,
                             verbose=verbose, **kwargs)
        logger.info("training finished")

    def _feature2records(self, idx, feature: Dict, mode: str) -> List[Dict]:
        record = dict(idx=idx, **feature)
        return [record]

    @log_cost_time
    def _model_infer(self, test_batches, model: Model = None, tf_serving_url=None, show_detail=False):

        if tf_serving_url:
            logger.info(f"infering with tf server:{tf_serving_url}...")
            pred_tensor_data = infer_with_tf_serving(test_batches,
                                                       tf_serving_url=tf_serving_url,
                                                       show_detail=show_detail)
        elif model:
            logger.info("infering with tf model...")
            pred_tensor_data = []
            test_batches = tqdm(test_batches) if show_detail else test_batches
            for batch in test_batches:
                pred_batch = model(batch, training=False)
                pred_tensor_data.extend(pred_batch)
            return pred_tensor_data

        else:
            raise Exception("neither nn_model or tf_serving_url are given!")
        return pred_tensor_data

    def infer(self, data, batch_size=32, show_detail=False, max_pred_num=None, tf_serving_url=None,
                overwrite_cache=False, **kwargs):
        logger.debug("infering with kwargs:")
        logger.debug(jdumps(dict(batch_size=batch_size, show_detail=show_detail, max_pred_num=max_pred_num, **kwargs)))
        data_manager = DataManager.get_instance(model=self, data=data)
        data_manager.store_features(overwrite_cache=overwrite_cache)
        preds = self._infer(data_manager=data_manager, batch_size=batch_size, show_detail=show_detail,
                              max_pred_num=max_pred_num, tf_serving_url=tf_serving_url,
                              **kwargs)
        return preds

    def _infer(self, data_manager: DataManager, batch_size, show_detail, max_pred_num,
                 tf_serving_url=None, **kwargs) -> List:
        test_batches = data_manager.get_test_batches(batch_size=batch_size, max_num=max_pred_num)
        pred_tensors = self._model_infer(test_batches=test_batches, model=self.nn_model,
                                           tf_serving_url=tf_serving_url, show_detail=show_detail)
        features = data_manager.get_features()
        preds = self._post_infer(features=features, pred_tensors=pred_tensors, show_detail=show_detail, **kwargs)
        return preds

    @abstractmethod
    @log_cost_time
    def _post_infer(self, features, pred_tensors, show_detail, threshold, **kwargs) -> List:
        pass

    def get_dataset_info(self, mode):
        if mode not in self.model_dict:
            raise ValueError(f"no model for mode:{mode}")
        model_info = self.model_dict[mode]
        return model_info["type_info"], model_info["shape_info"]

    def _update_model_dict(self, mode: str, model: Model):
        input_layers = [l for l in model.layers if isinstance(l, InputLayer)]
        type_info = dict([(l.name, l.dtype) for l in input_layers])
        shape_info = dict([(l.name, l.input_shape[0][1:]) for l in input_layers])
        model_info = dict(model=model, type_info=type_info, shape_info=shape_info)
        self.model_dict[mode] = model_info


class SingleScope(object):
    def __enter__(self):
        # logger.info("enter single scope")
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # logger.info("leave single scope")
        pass
