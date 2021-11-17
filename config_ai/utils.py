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
import logging
import collections
from configparser import ConfigParser
from snippets.utils import *
logger = logging.getLogger(__name__)
#
#
# # 创建一个目录
# def create_dir_path(path: str):
#     dir_path = os.path.dirname(path)
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#

# def pretty_floats(obj, r=4):
#     if isinstance(obj, float):
#         return round(obj, r)
#     elif isinstance(obj, dict):
#         return dict((k, pretty_floats(v)) for k, v in obj.items())
#     elif isinstance(obj, (list, tuple)):
#         return map(pretty_floats, obj)
#     return obj
#
# # 输入同时适配dict和list of dict两种情况
# def adapt_single(data_field):
#     def wrapper(func):
#         @wraps(func)
#         def rs_func(*args, **kwargs):
#             data = kwargs.get(data_field)
#             is_single = False
#             if not isinstance(data, list):
#                 kwargs[data_field] = [data]
#                 is_single = True
#             res = func(*args, **kwargs)
#             if is_single:
#                 res = res[0]
#             return res
#
#         return rs_func
#
#     return wrapper
#
#
# # 限制数据条目数
# def get_limit_data(data, max_num: int):
#     for idx, item in zip(range(max_num), data):
#         yield item
#
#
#
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


# def find_all(content: str, to_find: str, overlap=False, ignore_case=False) -> List[Tuple]:
#     """
#     从$content中找到所有$to_find出现的span
#     Args:
#         content: 在哪个字符串中查找
#         to_find: 需要查找的对象
#         overlap: 查找的span是否可以重叠
#         ignore_case: 是否忽略case
#     Returns: 所有匹配的span的列表
#
#     """
#     rs_list = []
#     if ignore_case:
#         content = content.lower()
#         to_find = to_find.lower()
#     if not to_find:
#         return rs_list
#     text_len = len(to_find)
#
#     beg = 0
#     while True:
#         b = content.find(to_find, beg)
#         if b == -1:
#             return rs_list
#         e = b + text_len
#         rs_list.append((b, e))
#         beg = b + 1 if overlap else e
#     return rs_list
#

#
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
#

# # 将一个序列随机切分成两部分
# def random_split(seq: Sequence, rate: float) -> Tuple[Sequence, Sequence]:
#     tmp = copy.copy(seq)
#     random.shuffle(tmp)
#     idx = int(rate * len(tmp))
#     return tmp[:idx], tmp[idx:]
#
#
# def replace_none(val, default_val):
#     return default_val if val is None else val


#
# # 将seq随机重排之后切成n等分
# def nfold(seq: Sequence, n) -> List[Tuple[Sequence, Sequence]]:
#     random.shuffle(seq)
#     num, remain = divmod(len(seq), n)
#     folds = []
#     start = 0
#     for idx in range(n):
#         end = start + num
#         if idx < remain:
#             end += 1
#
#         folds.append(seq[start:end])
#         start = end
#     return folds
#
#
# def inverse_dict(d: dict, overwrite=False):
#     rs = dict() if overwrite else collections.defaultdict(list)
#
#     def update_rs(_k, _v):
#         if overwrite:
#             rs[_k] = _v
#         else:
#             rs[_k].append(_v)
#
#     for k, v in d.items():
#         if isinstance(v, list):
#             for ele in v:
#                 update_rs(ele, k)
#         else:
#             update_rs(v, k)
#     return dict(rs)
#
#
# def find_span(l, val):
#     spans = []
#     start = None
#
#     for idx, v in enumerate(l):
#         if v == val and start is None:
#             start = idx
#         if v != val and start is not None:
#             spans.append((start, idx))
#             start = None
#     if start:
#         spans.append((start, len(l)))
#     return spans
