# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_utils
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
from abc import abstractmethod
from typing import List, Dict, Iterable

import numpy as np
from snippets.decorators import log_cost_time
from snippets.utils import jdump_lines, jload_lines, get_batched_data
from tensorflow.python.data import Dataset
from tqdm import tqdm

from config_ai.utils import truncate_seq

logger = logging.getLogger(__name__)


def truncate_record(record: Dict, max_len: int, keys=["token_ids", "segment_ids", "tokens"], **kwargs) -> Dict:
    for k in keys:
        if k in record:
            record[k] = truncate_seq(seq=record[k], max_len=max_len, **kwargs)

    return record


def records2batches(records, data_shape, batch_size):
    for batch in get_batched_data(records, batch_size):
        padded_batch = dict()
        for key, shape in data_shape.items():
            items = np.array([item[key] for item in batch])
            if shape:
                max_len = max(len(e) for e in items)
                items = [np.pad(array=item, pad_width=(0, max_len - len(item))) for item in items]
                items = np.stack(items)
            padded_batch[key] = items
        yield padded_batch


class DataManager(object):
    def __init__(self, model):
        self.model = model
        self.features: List[Dict] = None

    @classmethod
    def get_instance(cls, model, data, **kwargs):
        if isinstance(data, str):
            return FileDataManager(model=model, data_file=data, **kwargs)
        if isinstance(data, Iterable):
            return IterableDataManager(model=model, examples=data, **kwargs)
        raise ValueError(f"invalid data type:{type(data)}")

    @abstractmethod
    def get_features(self, return_generator=True, max_num=None):
        raise NotImplementedError

    @log_cost_time
    @abstractmethod
    def store_features(self, **kwargs):
        raise NotImplementedError

    def get_padded_shapes(self, mode: str, **kwargs) -> Dict:
        return self.model.get_dataset_info(mode, **kwargs)[1]

    def get_records(self, mode, max_num=None, return_generator=True, **kwargs) -> Iterable[Dict]:
        def gen():
            features = self.get_features(max_num=max_num)
            for idx, feature in enumerate(features):
                records = self.model._feature2records(idx=idx, feature=feature, mode=mode, **kwargs)
                for record in records:
                    yield record

        if return_generator:
            return gen()
        return list(gen())


    def get_dataset(self, mode: str, max_num=None, **kwargs) -> Dataset:
        dataset_type, dataset_shape = self.model.get_dataset_info(mode, **kwargs)

        def gen():
            records = self.get_records(mode=mode, max_num=max_num, **kwargs)
            for record in records:
                rs = {k: record[k] for k, dtype in dataset_type.items()}
                yield rs

        dataset = Dataset.from_generator(generator=gen, output_types=dataset_type)
        return dataset

    def get_test_batches(self, batch_size, max_num=None):
        dataset_type, dataset_shape = self.model.get_dataset_info(mode="test")
        records = self.get_records(mode="test", max_num=max_num, return_generator=True)
        return records2batches(records, dataset_shape, batch_size)

    def get_train_dataset(self, batch_size=32, buffer_size=None, repeat=None, **kwargs) -> Dataset:
        # logger.info(f"getting train dataset with batch_size:{batch_size}, shuffle_buffer_size:{buffer_size}")
        dataset = self.get_dataset(mode="train", **kwargs)
        dataset_shape = self.get_padded_shapes("train")
        if buffer_size:
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        train_dataset = dataset.repeat(repeat). \
            padded_batch(batch_size, padded_shapes=dataset_shape).prefetch(buffer_size=1)
        return train_dataset


class IterableDataManager(DataManager):
    def __init__(self, model, examples):
        super().__init__(model=model)
        self.examples = examples

    def store_features(self, **kwargs):
        if not self.features:
            logger.debug("storing features to memory")
            self.features = [self.model.example2feature(e) for e in tqdm(self.examples)]

    def get_features(self, return_generator=True, max_num=None):

        if self.features:
            logger.debug("load features from memory")
            features = self.features[:max_num] if max_num else self.features
            gen = (e for e in features)
        else:
            logger.debug("generate features")
            examples = self.examples[:max_num] if max_num else self.examples
            gen = (self.model.example2feature(example) for example in examples)
        return gen if return_generator else list(gen)


class FileDataManager(DataManager):
    def __init__(self, model, data_file: str):
        super().__init__(model=model)
        self.data_file = data_file
        self.model = model
        self.cache_file = f"{self.data_file}_cache_{self.model.model_name}.{self.model.tokenizer.get_cache_tag()}"

    def store_features(self, mode="file", overwrite_cache=False):
        examples = self.model.jload_lines(self.data_file, return_generator=True)
        if mode == "file":
            if overwrite_cache or not os.path.exists(self.cache_file):
                logger.info(f"storing features to cache file:{self.cache_file}")
                features = (self.model.example2feature(example) for example in tqdm(examples))
                jdump_lines(features, self.cache_file, progbar=True)
        if mode == "memory" and not self.features:
            logger.info("storing features to memory")
            self.features = [self.model.example2feature(e) for e in tqdm(examples)]

    def get_features(self, return_generator=True, max_num=None) -> Iterable[Dict]:
        if self.features:
            logger.info("load features from memory")
            features = self.features[:max_num] if max_num else self.features
            gen = (e for e in features)
        elif os.path.exists(self.cache_file):
            # logger.info(f"load features from cache file:{self.cache_file}")
            gen = jload_lines(self.cache_file, max_data_num=max_num, return_generator=True)
        else:
            logger.info("generate features")
            examples = self.model.jload_lines(self.data_file, max_data_num=max_num, return_generator=True)
            gen = (self.model.example2feature(e) for e in examples)
        return gen if return_generator else list(gen)
