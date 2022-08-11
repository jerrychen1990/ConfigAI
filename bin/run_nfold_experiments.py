#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_nfold_experiments
   Description :
   Author :       chenhao
   date：          2021/8/4
-------------------------------------------------
   Change Activity:
                   2021/8/4:
-------------------------------------------------
"""
import copy

import fire
import logging
import os

from config_ai.utils import read_config, jload_lines, nfold, jdump_lines, jdumps

from run_batch_experiments import run_batch_experiments

logger = logging.getLogger(__name__)


def run_nfold_experiments(n, config_path, worker_num=2, use_history=True):
    base_config_path = config_path

    base_config = read_config(config_path)
    logger.info("base_config:")
    logger.info(jdumps(base_config))

    train_data_path = base_config["data_config"].get("train_data_path")
    labeled_data = []
    if train_data_path:
        labeled_data.extend(jload_lines(train_data_path))
    eval_data_path = base_config["data_config"].get("eval_data_path")
    if eval_data_path:
        labeled_data.extend(jload_lines(eval_data_path))
    logger.info(f"get {len(labeled_data)} data")
    if not labeled_data:
        logger.info("no labeled data found, exit!")

    nfolds = nfold(labeled_data, n)
    data_dir = os.path.join(os.path.dirname(train_data_path), f"{n}folds")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    configs = []
    for idx in range(n):
        train_data_path = os.path.join(data_dir, f"train-{idx}.jsonl")
        eval_data_path = os.path.join(data_dir, f"dev-{idx}.jsonl")
        if not os.path.exists(train_data_path) or not os.path.exists(eval_data_path):
            logger.info(f"dumping data to {train_data_path} and {eval_data_path}...")

            eval_data = nfolds[idx]
            train_data = [e for i in range(n) for e in nfolds[i] if i != idx]
            jdump_lines(train_data, train_data_path)
            jdump_lines(eval_data, eval_data_path)
        configs.append([f"fold-{idx}", {"data_config.train_data_path": train_data_path, "data_config.eval_data_path": eval_data_path}])

    logger.info(jdumps(configs))
    run_batch_experiments(base_config_path=base_config_path, configs=[configs], worker_num=worker_num, use_history=use_history)


if __name__ == '__main__':
    # run_nfold_experiments(n=5, config_path="/nfs/pony/chenhao/workspace/clue/afqmc/roberta_wwm_large.ini")
    fire.Fire(run_nfold_experiments)


"""
run_nfold_experiments.py \
--n=5 \
--config_path=/nfs/pony/chenhao/workspace/clue/afqmc/roberta_wwm_large.ini
"""