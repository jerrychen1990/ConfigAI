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
import collections
from configparser import ConfigParser
from snippets.utils import *
from snippets.decorators import *

logger = logging.getLogger(__name__)


# # 深度遍历用u更新d
def deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


#

# 读一个.ini配置文件路径。对其中的值做eval之后，得到dict格式的配置内容
def read_config(config_path: str) -> dict:
    def eval_param(param):
        if isinstance(param, str):
            try:
                if param.upper() == "TRUE":
                    return True
                if param.upper() == "FALSE":
                    return False
                rs = eval(param)
                return rs
            except Exception as e:
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
    if config_path.endswith(".json"):
        cfg_dict = jload(config_path)

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
