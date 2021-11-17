#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_experiment_yud
   Description :
   Author :       chenhao
   date：          2021/7/22
-------------------------------------------------
   Change Activity:
                   2021/7/22:
-------------------------------------------------
"""
import logging

import fire

from config_ai.utils import read_config, execute_cmd

logger = logging.getLogger(__name__)

cmd_pattern = '''yudctl run -t {tag} -g={g_num} -d=24h -m=40 \
-i=registry.cn-hangzhou.aliyuncs.com/eigenlab/config-ai:tf2.2 \
-p=/nfs/pony/chenhao/workspace/ConfigAI \
-u=chenhao \
-r=requirements.txt \
-y \
python bin/run_experiment.py {config_path}
'''


def run_experiment_yud(config_path):
    config = read_config(config_path)
    g_num = config["common_config"].get("g_num", 1)
    tag = f"{config['common_config']['project_name']}-{config['common_config']['model_name']}"
    kwargs = dict(g_num=g_num, tag=tag, config_path=config_path)
    cmd = cmd_pattern.format(**kwargs)
    execute_cmd(cmd)


if __name__ == '__main__':
    fire.Fire(run_experiment_yud)
