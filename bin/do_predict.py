#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     do_predict
   Description :
   Author :       chenhao
   date：          2021/4/25
-------------------------------------------------
   Change Activity:
                   2021/4/25:
-------------------------------------------------
"""
import logging
import itertools
import fire

from config_ai.models import get_model_class_by_name
from config_ai.models.core import AIConfigBaseModel
from config_ai.utils import get_batched_data, jdump_lines


logger = logging.getLogger(__name__)


def do_predict(model_cls, ckpt_path, data_path, output_path,
               batch_size: int = 1024, test_kwargs="{}", continue_idx=0):
    logger.info("test_kwargs")
    logger.info(test_kwargs)

    model_cls = get_model_class_by_name(model_cls)
    logger.info(f"loading model from :{ckpt_path}")
    model: AIConfigBaseModel = model_cls.load(path=ckpt_path)
    data = model.jload_lines(data_path = data_path, return_generator=True)
    # logger.info(f"got {len(data)} data from {data_path}")
    logger.info(type(batch_size))
    logger.info(batch_size)

    batch_size = int (batch_size)
    continue_idx = int(continue_idx)


    if continue_idx:
        logger.info(f"continue from:{continue_idx}")
        data = itertools.islice(data,continue_idx, None)
    else:
        with open(output_path, "w") as f:
            pass
    batches = get_batched_data(data, batch_size)
    for idx, batch in enumerate(batches):
        logger.info(type(idx))
        logger.info(type(continue_idx))

        logger.info(f"predicting data batch:{idx}, batch_size:{len(batch)},origin_idx:{continue_idx+idx*batch_size}")
        preds = model.predict(batch, **test_kwargs)
        output_data = []
        logger.info(f"generating output_data")
        for item, pred in zip(batch, preds):
            tmp = item.dict()
            tmp.update(predict=pred)
            output_data.append(tmp)

        logger.info(f"dumping {len(output_data)} output data to path:{output_path}")
        jdump_lines(obj=output_data, fp=output_path, mode="a")
    logger.info("job finished!")

if __name__ == "__main__":
    fire.Fire(do_predict)

"""
do_predict.py \
--model_cls=TFTextClassifyModel \
--ckpt_path=/nfs/pony/chenhao/experiment/clue_tnews/roberta_wwm_base/model \
--data_path=/nfs/pony/chenhao/data/clue/tnews/test.jsonl \
--output_path=/nfs/pony/chenhao/experiment/clue_tnews/roberta_wwm_base/output/test.json \
--test_kwargs={"batch_size":32}
"""



