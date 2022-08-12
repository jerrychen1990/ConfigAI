#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_batch_experiments
   Description :
   Author :       chenhao
   date：          2021/6/24
-------------------------------------------------
   Change Activity:
                   2021/6/24:
-------------------------------------------------
"""
import copy
import logging
import subprocess
import time
from itertools import product
from typing import List, Dict

import fire

from config_ai.utils import jdumps, read_config, jdump, jload

logger = logging.getLogger(__name__)


def update_config(cfg, path, v):
    keys = path.split(".")
    d = cfg
    for key in keys[:-1]:
        d = d.get(key, dict())
    d[keys[-1]] = v


cmd_pattern = '''yudctl run -t {tag} -g={g_num} -d=24h -m=40 \
-i=registry.cn-hangzhou.aliyuncs.com/eigenlab/config-ai:tf2.2 \
-p=${CONFIG_AI_PATH} \
-u=chenhao \
-r=requirements.txt \
-y \
python bin/run_experiment.py {config_path}
'''


def run_batch_experiments(base_config_path: str, configs: List[Dict], worker_num: int, use_history=True, repeat=1):
    logger.info(f'''
        running batch_experiments...
        base_config_path:{base_config_path}
        worker_num:{worker_num}
    ''')
    logger.info(f"batch experiment configs:\n{jdumps(configs)}")

    base_config = read_config(base_config_path)
    # logger.info(f"base config:{jdumps(base_config)}")

    model_name = base_config["common_config"].get("model_name", "tf")

    cmds = []
    # experiment_num = reduce(lambda x, y: x * len(y), configs, 1)
    for items in product(*configs):
        for idx in range(repeat):
            logger.info(f"repeat:{idx}")
            logger.info(items)
            suffix = "_".join(e[0] for e in items)
            m_name = model_name + "_" + suffix if model_name else suffix
            config = copy.deepcopy(base_config)
            config["common_config"]["model_name"] = m_name
            tag = config["common_config"]["project_name"] + "_" + m_name
            if repeat >1 :
                tag += "_" + str(idx)
            for info in [e[1] for e in items]:
                for k, v in info.items():
                    logger.info(f"update {k} to {v}")
                    update_config(config, k, v)
            # logger.info(jdumps(config))
            g_num = config["common_config"].get("g_num", 1)
            tmp_config_path = f"/nfs/tmp/config_ai/{tag}.json"
            jdump(config, tmp_config_path)
            cmd = cmd_pattern.format(tag=tag, g_num=g_num, config_path=tmp_config_path)
            logger.info(f"adding job:{tag} to queue")
            cmds.append((tag, cmd))

    cmds = cmds[::-1]
    if use_history:
        logger.info("check history status")
        tmp_cmds = []
        for tag, cmd in cmds:
            status = get_status(tag)
            logger.info(f"job {tag}'s history status:{status}")
            if status not in ["Succeeded", "Running", "Pending"]:
                tmp_cmds.append((tag, cmd))
        cmds = tmp_cmds

    cmds_num = len(cmds)
    logger.info(f"got {cmds_num} experiments to run")

    working_tags = []
    interval = 60
    interval_idx = 0
    end_num = 0

    while True:
        logger.info(f"checking working {len(working_tags)} job's status ...")

        tmp_working_tags = []
        for idx, tag in enumerate(working_tags):
            status = get_status(tag)
            logger.info(f"job {tag} current status:{status}")
            if status in ["Succeeded", "Failed", "Stopped"]:
                logger.info(f"job {tag} ends")
                end_num += 1
            else:
                tmp_working_tags.append(tag)
        working_tags = tmp_working_tags

        remain_num = cmds_num - end_num - len(working_tags)
        running_num = len(working_tags)
        logger.info(f"{interval_idx * interval} seconds parsed, {end_num} done, {running_num} running,"
                    f" {remain_num} remaining")
        if remain_num + running_num<=0:
            break


        logger.info("submitting new job ...")

        while cmds and len(working_tags) < worker_num:
            tag, cmd = cmds.pop()
            config = jload(f"/nfs/tmp/config_ai/{tag}.json")
            logger.info(f"submit job: {tag} with config:\n{jdumps(config)}")
            # jdump(config, tmp_config_path)
            # logger.info(cmd)
            status, output = execute_cmd(cmd)
            if status == 0:
                logger.info(f"job {tag} submit successfully")
                working_tags.append(tag)

        logger.info(f"wait {interval} seconds")
        interval_idx += 1
        time.sleep(interval)


def run_batch_experiments_config(config_path):
    config = read_config(config_path)
    run_batch_experiments(**config)


def get_status(tag):
    cmd = "yudctl list"
    s, output = execute_cmd(cmd)
    status_list = output.split("\n")[3:-1]
    status_list = [[t.strip() for t in e.split("|") if t] for e in status_list]
    status_list = [e for e in status_list if e[1] == tag]
    if status_list:
        return status_list[0][2]
    return "Unknown"


def execute_cmd(cmd):
    logger.info(f"execute command:\n{cmd}")
    status, output = subprocess.getstatusoutput(cmd)
    return status, output


if __name__ == '__main__':
    fire.Fire(run_batch_experiments_config)

"""
python run_batch_experiments.py /nfs/pony/chenhao/workspace/clue/wsc/batch_experiments.json

"""
