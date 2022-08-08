# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     experiments
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""
import random
from abc import ABCMeta, abstractmethod

# from config_ai.callbacks import Evaluator
from config_ai.evaluate import *
from config_ai.models import *
from config_ai.models.core import AIConfigBaseModel
from config_ai.schema import *
from config_ai.utils import print_info, jdumps, get_current_time_str, jdump

logger = logging.getLogger(__name__)


def star_print(info):
    return print_info(info, target_logger=logger, fix_length=128)


def get_model_config(experiment_config):
    config_keys = ['tokenizer_config', 'task_config']
    rs_config = {k: experiment_config.get(k) for k in config_keys}
    rs_config.update(model_name=experiment_config["common_config"]["model_name"])
    return rs_config


# 基础实验类
class BaseExperiment(metaclass=ABCMeta):
    # 通过config初始化实验
    valid_models = []

    def __init__(self, config: dict):
        # set_tf_config()
        star_print("experiment param:")
        logger.info(jdumps(config))
        self.model: Optional[AIConfigBaseModel] = None
        self.config = config
        # load common config
        self.common_config: dict = config['common_config']
        self.model_cls = get_model_class_by_name(self.common_config["model_cls"])
        assert self.model_cls in self.valid_models
        random_seed = self.common_config.get("default_random_seed", random.randint(0, 1e9))
        # set_random_seed(random_seed)
        self.common_config["random_seed"] = random_seed
        self.experiment_dir = self.common_config['experiment_dir']
        self.project_name = self.common_config['project_name']
        self.model_name = self.common_config['model_name']
        self.ckpt_path = self.common_config.get("ckpt_path")
        self.is_train = self.common_config['is_train']
        self.is_test = self.common_config.get('is_test', False)
        self.is_save = self.common_config['is_save']
        self.save_args = self.common_config.get('save_args', {})
        self.is_overwrite_experiment = self.common_config.get("is_overwrite_experiment", True)
        self.output_phase_list = self.common_config['output_phase_list']
        self.eval_phase_list = self.common_config['eval_phase_list']
        # init experiment path
        self.project_dir = os.path.join(self.experiment_dir, self.project_name)
        if "experiment_path" in self.common_config.keys():
            self.experiment_path = self.common_config["experiment_path"]
        elif self.is_overwrite_experiment:
            self.experiment_path = os.path.join(self.project_dir, self.model_name)
        else:
            self.experiment_path = os.path.join(self.project_dir, self.model_name, get_current_time_str())

        self.tensorboard_dir = os.path.join(self.project_dir, "tensorboard")
        self.eval_dir = os.path.join(self.experiment_path, "eval")
        self.output_dir = os.path.join(self.experiment_path, "output")
        self.log_dir = os.path.join(self.experiment_path, "log")
        self.model_path = os.path.join(self.experiment_path, "model")
        self.ckpt_dir = os.path.join(self.experiment_path, "ckpt")

        # load data config
        self.data_config = self.config['data_config']
        self.train_data_path = self.data_config.get('train_data_path')
        self.dev_data_path = self.data_config.get('dev_data_path')
        self.test_data_path = self.data_config.get('test_data_path')
        # load nn model config
        self.nn_model_config: dict = config.get('nn_model_config', {})
        # load compile config
        self.compile_config: Optional[dict] = config.get('compile_config')
        # load train config
        self.train_config: Optional[dict] = config.get('train_config')
        # load test config
        self.test_config: Optional[dict] = config.get('test_config')
        # load callback config
        self.callback_config: Optional[dict] = config.get("callback_config")

    # 初始化要续联的模型
    def initialize_model(self):
        star_print("model initialize phase start")
        # 加载ckpt中的模型
        if self.ckpt_path:
            star_print("initialize nn_model from checkpoint")
            self.model = self.model_cls.load(path=self.ckpt_path, load_type="json", load_nn_model=True)
        # 根据配置，新建一个模型
        else:
            star_print("initialize nn_model from nn_model config")
            model_config = get_model_config(self.config)
            self.model = self.model_cls(config=model_config)
            self.model.build_model(**self.nn_model_config)
        star_print("model initialize phase end")

    # 根据配置，获得所有的keras callback方法
    def _get_callbacks(self):
        logger.info("initializing callbacks...")
        callbacks = []
        # tf支持的所有callback
        is_tensorboard_callback = self.callback_config.get("tensorboard_callback", False)
        # # 写入tensorboard的callback方法
        # if is_tensorboard_callback:
        #     tensorboard_log_dir = os.path.join(self.tensorboard_dir,
        #                                        f"{self.model_name}-{get_current_time_str()}")
        #     tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, update_freq=200)
        #     logger.info(f"add tensorboard callback with log path:{tensorboard_log_dir}")
        #     callbacks.append(tensorboard)
        # # 当dev loss不再下降时，提前终止训练的方法
        # early_stop_kwargs = self.callback_config.get("early_stop_kwargs")
        # if early_stop_kwargs is not None:
        #     logger.info(f"add early stop callback with kwargs:{early_stop_kwargs}")
        #     early_stop = tf.keras.callbacks.EarlyStopping(**early_stop_kwargs)
        #     callbacks.append(early_stop)
        # epoch结束时测评模型结果的callback
        # evaluator_kwargs = self.callback_config.get("evaluator_kwargs")
        # if evaluator_kwargs is not None:
        #     evaluator = Evaluator(experiment=self, **evaluator_kwargs)
        #     callbacks.append(evaluator)
        logger.info(f"will train with callbacks:{callbacks}")
        return callbacks

    # 训练模型
    def train_model(self):
        star_print("training phase start")
        assert self.model is not None
        self.model.compile_model(**self.compile_config)
        callbacks = self._get_callbacks()
        self.model.train(train_data=self.train_data_path,
                         dev_data=self.dev_data_path,
                         callbacks=callbacks,
                         **self.train_config)
        star_print("training phase end")

    # 保存模型
    def save_model(self):
        star_print("saving phase start")
        self.model.save(path=self.model_path, **self.save_args)
        star_print("saving phase end")

    # 测试模型
    def test_model(self):
        star_print("testing phase start")
        for tag, data_path in zip(['train', 'dev', 'test'],
                                  [self.train_data_path, self.dev_data_path, self.test_data_path]):
            if tag not in self.eval_phase_list and tag not in self.output_phase_list:
                continue
            logger.info("infer result on {} data:".format(tag))
            preds = self.model.infer(data=data_path, **self.test_config)
            examples = self.model.jload_lines(data_path, return_generator=True)
            output_data = self.get_output(examples, preds)
            path = os.path.join(self.output_dir, f"{tag}.json")
            logger.info(f"output pred :{len(output_data)} result to {path}")
            jdump(output_data, path)
            if tag in self.eval_phase_list:
                logger.info("evaluating on {} data".format(tag))
                examples = self.model.jload_lines(data_path, return_generator=True)
                eval_rs = self.evaluate(examples, preds)
                logger.info(jdumps(eval_rs))
                path = os.path.join(self.eval_dir, f"{tag}.json")
                logger.info("writing eval result to :{}".format(path))
                jdump(eval_rs, path)
        star_print("testing phase end")

    # 记录实验运行时的config
    def save_config(self):
        config_file_path = os.path.join(self.experiment_path, "config.json")
        star_print(f"saving config file to {config_file_path}")
        jdump(self.config, config_file_path)

    # 运行实验，实验的入口
    def run(self):
        star_print("experiment start")
        self.save_config()
        self.initialize_model()
        if self.is_train:
            self.train_model()
        if self.is_save:
            self.save_model()
        if self.is_test:
            self.test_model()
        star_print("experiment end")

    # 评测模型效果，不同种类的实验需要不同的评测实现
    @abstractmethod
    def evaluate(self, examples: List, preds: List) -> Dict:
        raise NotImplementedError

    # 输出模型预测结果，不同种类的实现需要不同的输出实现
    @abstractmethod
    def get_output(self, examples: List, preds: List) -> Dict:
        raise NotImplementedError
#
#
# class TextClassifyExperiment(BaseExperiment):
#     valid_models = [CLSTokenClassifyModel, MLMTextClassifyModel]
#
#     def evaluate(self, examples: List[LabeledTextClassifyExample], preds: List[LabelOrLabels]) -> Dict:
#         true_labels = [e.label for e in examples]
#         rs = eval_text_classify(true_labels, preds)
#         return rs
#
#     def get_output(self, examples: List[UnionTextClassifyExample], preds: List[LabelOrLabels]) -> List[dict]:
#         return get_text_classify_output(examples, preds)
#
#
# #
# class TextSpanClassifyExperiment(BaseExperiment):
#     valid_models = [SeqLabelingModel, GlobalPointerModel]
#
#     def evaluate(self, examples: List[LabeledTextSpanClassifyExample], preds: List[TextSpans]) -> Dict:
#         true_text_spans = [e.text_spans for e in examples]
#         rs = eval_text_span_classify(true_text_spans, preds)
#         return rs
#
#     def get_output(self, examples: List[UnionTextSpanClassifyExample], preds: List[TextSpans]) -> List[dict]:
#         return get_text_span_classify_output(examples, preds)
#
#
# #
# class RelationClassifyExperiment(BaseExperiment):
#     valid_models = [RelationTokenClassifyModel]
#
#     def evaluate(self, examples: List[LabeledRelationClassifyExample], preds: List[LabelOrLabels]) -> Dict:
#         true_labels = [e.label for e in examples]
#         rs = eval_relation_classify(true_labels, preds)
#         return rs
#
#     def get_output(self, examples: List[UnionRelationClassifyExample], preds: List[LabelOrLabels]) -> List[dict]:
#         return get_relation_classify_output(examples, preds)
#
#
# #
# class MLMExperiment(BaseExperiment):
#     valid_models = [TransformerMLMModel]
#
#     def evaluate(self, examples: List[MLMExample], preds: List[List[str]]) -> Dict:
#         masked_tokens_list = [e.masked_tokens for e in examples]
#         rs = eval_mlm(masked_tokens_list, preds)
#         return rs
#
#     def get_output(self, examples: List[MLMExample], preds: List[List[str]]) -> List[dict]:
#         return get_mlm_output(examples, preds)
#
#
# #
# #
# # class SPOExtractExperiment(BaseExperiment):
# #     def evaluate(self, output_data: List[SPOExtractExample]) -> Dict:
# #         true_target_lists: List[List[SPO]] = [e.true_infer for e in output_data]
# #         pred_target_lists: List[List[SPO]] = [e.extra_info['infer'] for e in output_data]
# #         rs = eval_spo_extract_result(true_target_lists, pred_target_lists)
# #         return rs
# #
# #     def get_output(self, examples: List[SPOExtractExample], preds: List[List[SPO]]) -> List[SPOExtractExample]:
# #         return get_spo_extract_output(examples, preds)
# #
# #
# class Seq2SeqExperiment(BaseExperiment):
#     valid_models = [TransformerSeq2SeqModel]
#
#     def evaluate(self, examples: List[LabeledSeq2SeqExample], preds: List[List[GenText]]) -> Dict:
#         tgt_texts: List[GenText] = [e.tgt_text for e in examples]
#         rs = eval_seq2seq(tgt_texts, preds)
#         return rs
#
#     def get_output(self, examples: List[UnionSeq2SeqExample], preds: List[List[GenText]]) -> List[dict]:
#         return get_seq2seq_output(examples, preds)
#
#
# class ExperimentFactory:
#     _EXPERIMENTS = [TextClassifyExperiment, TextSpanClassifyExperiment, RelationClassifyExperiment, MLMExperiment,
#                     Seq2SeqExperiment]
#
#     _MODEL2EXPERIMENT = {model: experiment for experiment in _EXPERIMENTS for model in experiment.valid_models}
#
#     @classmethod
#     def create(cls, config_path: str):
#         config = read_config(config_path)
#         model_cls = config["common_config"].get("model_cls")
#         model_cls = get_model_class_by_name(model_cls)
#         builder = cls._MODEL2EXPERIMENT.get(model_cls)
#         if not builder:
#             raise ValueError(
#                 f"not valid model_cls:{model_cls}, "
#                 f"valid model_cls list:{list(cls._MODEL2EXPERIMENT.keys())}"
#             )
#         return builder(config)
