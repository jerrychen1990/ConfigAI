# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     callbacks
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""
import json
import logging
import math
import os

import tensorflow as tf
from jsonpath_ng import parse

from config_ai.models.tf_core import TFBasedModel
from snippets import jdump, pretty_floats

logger = logging.getLogger(__name__)


class Evaluator(tf.keras.callbacks.Callback):
    def __init__(self,
                 experiment,
                 epoch_freq=None,
                 step_freq=None,
                 monitor=None,
                 mode="MAX",
                 save_mode="BEST",
                 restore_best_weights=True,
                 verbose=0,
                 **kwargs):
        """
        在epoch结束时用实验的dev集合做测评的callback
        Args:
            experiment: 添加该evaluator的实验
            monitor: 对比指标在测评结果中的json_path
            mode: MAX/MIN 测评指标越大越好还是越小越好
            save_mode: ALWAYS/BEST/NEVER 模型保存的模型，
                ALWAYS：不管测评结果如何，都要保存模型
                BEST：只在测评结果达到最好时保存模型
                NEVER：从不保存模型
            step_freq: 每几个step做一次测评
            reload_best_weight: 在训练结束时，是否load测评结果最好的一次模型weigh
            verbose: 是否输出细节信息,0:不输出，1:输出
        """
        super().__init__()
        self.experiment = experiment
        self.experiment_model: TFBasedModel = experiment.model
        self.ckpt_dir = self.experiment.ckpt_dir
        self.dev_data_path = self.experiment.dev_data_path
        self.test_config = self.experiment.test_config
        self.train_batch_size = self.experiment.train_config["batch_size"]
        self.dev_data = self.experiment_model.jload_lines(self.dev_data_path)
        self.monitor = monitor
        self.mode = mode
        self.save_mode = save_mode
        self.epoch_freq = epoch_freq
        self.step_freq = step_freq
        self.restore_best_weights = restore_best_weights
        self.best_metric = 0. if self.mode.upper() == "MAX" else math.INF
        self.best_weights = None
        self.cur_step = 0
        self.cur_epoch = 0
        self.verbose = verbose
        self.history_path = os.path.join(self.experiment.experiment_path, "history.txt")
        with open(self.history_path, "w") as f:
            pass

    def _is_better(self, metric: float) -> bool:
        if metric > self.best_metric and self.mode.upper() == "MAX":
            return True
        if metric < self.best_metric and self.mode.upper() == "MIN":
            return True
        return False

    def _save_model(self, output=None, eval_rs=None):
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.cur_step}")
        ckpt_model_path = os.path.join(ckpt_path, "model")
        logger.info(f"current step:{self.cur_step}, saving ckpt to {ckpt_model_path}")
        self.experiment_model.save(path=ckpt_model_path)
        if output:
            output_path = os.path.join(ckpt_path, "output.json")
            logger.info(f"saving output to {output_path}")
            jdump(output, output_path)
        if eval_rs:
            eval_path = os.path.join(ckpt_path, "eval.json")
            logger.info(f"saving eval rs to {eval_path}")
            jdump(eval_rs, eval_path)

    def _handle_ckpt(self, logs):
        if self.monitor:
            logger.info(f"evaluate on dev data for step:{self.cur_step}")

            pred_data = self.experiment_model.predict(data=self.dev_data_path,
                                                      **self.test_config)
            output_data = self.experiment.get_output(self.dev_data, pred_data)
            eval_rs = self.experiment.evaluate(self.dev_data, pred_data)
            expected_expr = parse(self.monitor)
            metric = [a.value for a in expected_expr.find(eval_rs)][0]

            logger.info(f"history path: {self.history_path}")

            cur_info = dict(epoch=self.cur_epoch, step=self.cur_step, example=self.cur_step * self.train_batch_size, metric=metric, **logs)

            with open(self.history_path, "a") as history_file:
                cur_info = pretty_floats(cur_info)
                history_file.write(json.dumps(cur_info, ensure_ascii=False))
                history_file.write("\n")

            logger.info(f"get metric value:{metric:6.5f} with monitor:{self.monitor}")

            if self._is_better(metric):
                logger.info("got best metric!")
                self.best_metric = metric
                if self.restore_best_weights:
                    logger.info("saving best weights...")
                    self.best_weights = self.model.get_weights()
                if self.save_mode.upper() != 'NEVER':
                    self._save_model(output=output_data, eval_rs=eval_rs)
            elif self.save_mode.upper() == 'ALWAYS':
                self._save_model(output=output_data, eval_rs=eval_rs)
        else:
            if self.save_mode.upper() == 'ALWAYS':
                self._save_model()

    def on_batch_end(self, batch, logs=None):
        self.cur_step += 1
        if self.step_freq and self.cur_step % self.step_freq == 0:
            self._handle_ckpt(logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.cur_epoch += 1
        if self.epoch_freq and self.cur_epoch % self.epoch_freq == 0:
            self._handle_ckpt(logs=logs)

    def on_train_end(self, logs=None):
        if self.step_freq and self.cur_step % self.step_freq != 0:
            self._handle_ckpt(logs=logs)

        if self.epoch_freq and self.cur_epoch % self.epoch_freq != 0:
            self._handle_ckpt(logs=logs)

        if self.restore_best_weights and self.best_weights:
            if self.verbose > 0:
                logger.info('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)
