# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       chenhao
   date：          2021/3/30
-------------------------------------------------
   Change Activity:
                   2021/3/30:
-------------------------------------------------
"""
import re

from configparser import ConfigParser
from dataclasses import field, fields

import yaml
from snippets.decorators import *
from snippets.utils import *

logger = logging.getLogger(__name__)


# # 深度遍历用u更新d
def deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def eval_env(text):
    pattern = "\$\{.*?\}"
    for item in re.findall(pattern, text):
        text = text.replace(item, os.environ[item[2:-1]])
    return text


#

# 读取配置文件，支持.json/.ini/.yaml格式
# 可以继承另一个配置文件
# 可以引入环境变量
def read_config(config_path: str) -> dict:
    def eval_param(param):
        if isinstance(param, str):
            try:
                if param.upper() == "TRUE":
                    return True
                if param.upper() == "FALSE":
                    return False
                param = eval_env(param)
                param = eval(param)
                return param
            except:
                return param
        if isinstance(param, dict):
            return {k: eval_param(v) for k, v in param.items()}
        if isinstance(param, list):
            return [eval_param(v) for v in param]
        return param

    # convert cfg data to dict
    def cfg2dict(cfg):
        sections = cfg.sections()
        rs = {k: dict(cfg[k]) for k in sections}
        return rs

    cfg_dict = dict()
    if not os.path.exists(config_path):
        raise Exception(f"file {config_path} not exists!")

    logger.info(f"parsing config with path:{config_path}")
    if config_path.endswith(".ini"):
        parser = ConfigParser()
        parser.read(config_path)
        cfg_dict = cfg2dict(parser)
    elif config_path.endswith(".json"):
        cfg_dict = jload(config_path)
    elif config_path.endswith(".yaml"):
        with open(config_path, mode='r', encoding="utf") as stream:
            cfg_dict = yaml.safe_load(stream)
    else:
        raise ValueError(f"invalid config path:{config_path}")
    cfg_dict = eval_param(cfg_dict)

    if cfg_dict.get("common_config", {}).get("base_config"):
        logger.info("loading base config...")
        base_cfg_dict = read_config(cfg_dict["common_config"]["base_config"])
    else:
        base_cfg_dict = dict()
    deep_update(base_cfg_dict, cfg_dict)
    return base_cfg_dict


# 截断序列
def truncate_seq(seq: Sequence, max_len: int, mode="tail", keep_head=True, keep_tail=True):
    if not seq:
        return seq

    if mode == "tail":
        if keep_tail:
            return seq[:-1][:max_len - 1] + [seq[-1]]
        return seq[:max_len]
    if mode == "head":
        if keep_head:
            return [seq[0]] + seq[1:][-(max_len - 1):]
        return seq[-max_len:]
    raise ValueError(f"not valid mode:{mode}!")


def inverse_dict(d: dict, overwrite=False):
    rs = dict() if overwrite else collections.defaultdict(list)

    def update_rs(_k, _v):
        if overwrite:
            rs[_k] = _v
        else:
            rs[_k].append(_v)

    for k, v in d.items():
        if isinstance(v, list):
            for ele in v:
                update_rs(ele, k)
        else:
            update_rs(v, k)
    return dict(rs)


def safe_build_data_cls(cls, kwargs):
    _fields = fields(cls)
    valid_keys = set(e.name for e in _fields)
    kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    return cls(**kwargs)


def find_span(l, val):
    spans = []
    start = None

    for idx, v in enumerate(l):
        if v == val and start is None:
            start = idx
        if v != val and start is not None:
            spans.append((start, idx))
            start = None
    if start:
        spans.append((start, len(l)))
    return spans


def dlist2ldict(d_list):
    keys = d_list[0].keys
    rs = {k: list() for k in keys()}
    for e in d_list:
        for k, v in e.items():
            rs[k].append(v)
    return rs
