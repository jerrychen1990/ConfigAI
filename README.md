# ConfigAI

顾名思义，ConfigAI是一个基于配置的AI模型训练框架，目标是让训练AI模型像写配置文件一样简单，做到0门槛训练模型😊

ConfigAI包含如下功能

1. 定义标准化模型。将常见的模型分为以下几类，每类模型都有统一的输入输出格式(
   详见[configai_schema](https://github.com/jerrychen1990/ConfigAI/blob/main/config_ai/schema.py))：
    - text_classify(文本分类)
    - text_span_classify(mention识别,比如NER)
    - relation_classify(给定文本以及文本中的两个判断，做关系分类。常用于知识图谱构建)
2. 一些工具脚本，实现模型训练、错误分析、参数搜索 3一些notebook，帮助理解、调试ConfigAI的内部实现

## 内容目录

- [QuickStart](##QuickStart)
- [标准化模型](##标准化模型)
- [notebooks](##notebooks)

## QuickStart

执行下面步骤，快速实现一个NER模型的训练！

1. 准备训练/验证/测试数据集
    - 数据格式需要满足[configai_schema](https://github.com/jerrychen1990/ConfigAI/blob/main/config_ai/schema.py)中 TextSpanClassifyExample的规定
    - 以jsonline格式放置在磁盘中下
2. 准备配置文件
    - 以.ini或者.json格式填写实验配置
    - 配置文件包含整个训练、预测、评测、保存阶段的参数
    - 配置文件可以继承另一个配置文件，base_config字段表示父配置的路径，避免重复配置
    - 配置[示例](examples/text_span_classify/ner_seq_labeling.ini)
3. 执行运行实验脚本，将配置文件路径作为参数传入
    - ```shell
         python bin/run_experiment.py --config_path=examples/text_classify/sentiment_cls_token_classify.ini
      ```
    - 实验输出路径在{experiment_dir}/{project}/{model_name}目录下，也可以在实验日志里找到输出路径
## 标准化模型
### 通用配置
- 配置支持.ini格式和.json格式
- 配置支持继承,同名的key会用子配置的值覆盖父配置的值
- [示例配置](examples/base_config_bak.ini)

## notebooks
展示ConfigAI的内部实现，调试时使用
