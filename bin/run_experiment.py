#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_experiment
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""

import os
import logging
import traceback

import shutil

import fire

from config_ai.experiments import ExperimentFactory

logger = logging.getLogger(__name__)


def run_experiment(config_path):
    experiment = ExperimentFactory.create(config_path=config_path)
    try:
        experiment.run()
    except Exception as e:
        logger.error("run experiment failed, clean experiment files...")
        experiment_path = experiment.experiment_path
        if os.path.exists(experiment_path):
            logger.info(f"removing failed experiment :{experiment_path}")
            shutil.rmtree(experiment_path)
        logger.exception("")


if __name__ == '__main__':
    fire.Fire(run_experiment)
