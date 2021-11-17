# ConfigAI
顾名思义，ConfigAI是一个基于配置的AI模型训练框架，目标是让训练AI模型像写配置文件一样简单，做到0门槛训练模型😊

ConfigAI包含如下功能
1. 定义标准化模型。将常见的模型分为以下几类，每类模型都有统一的输入输出格式(详见[ai_schema](https://git.aipp.io/EigenLab/ai-schema))：
    - text_classify(文本分类)
    - text_span_classify(mention识别,比如NER)
2. 自动构建标准化模型训练的pony插件
3. 自动构建标准化模型的EulerService镜像
4. 一些工具脚本，实现参数搜索、模型融合等功能
5. 一个图形界面，实现模型上传、错误分析等功能
6. 一些notebook，帮助理解、调试ConfigAI的内部实现

## 内容目录

- [QuickStart](##QuickStart)
- [标准化模型](##标准化模型)
- [pony插件](##pony插件)
- [Euler服务](##Euler服务)
- [工具脚本](##工具脚本)
- [图形界面](##图形界面)
- [notebooks](##notebooks)



## QuickStart
执行下面步骤，快速实现一个NER模型的训练和部署！
1. 准备训练/验证/测试数据集
    - 数据格式需要满足[ai_schema](https://git.aipp.io/EigenLab/ai-schema)中 TextSpanClassifyExample的规定
    - 以jsonline格式放置在NFS路径下
    - 例如：/nfs/pony/chenhao/data/clue/cluener/train.jsonl
2. 准备配置文件
    - 以.ini或者.json格式放置在NFS路径下
    - 配置文件包含整个训练、预测、评测、保存阶段的参数
    - 配置文件可以继承另一个配置文件，base_config字段表示父配置的路径，避免重复配置
    - 配置[示例](examples/text_span_classify/tf_seq_labeling_config.ini)
3. 运行pony实验，将配置文件路径作为参数传入
    - 所有插件在pony的：非标准插件/ConfigAI目录下
    - [示例插件](https://pony.aidigger.com/plugin/detail/1191/version/0)
    - [示例实验](https://pony.aidigger.com/experiment/update/7337)
    - 实验输出路径在{experiment_dir}/{project}/{model_name}目录下，也可以在pony日志里找到输出路径
4. 部署euler服务
    - 基于ideal的服务[示例](https://euler.aidigger.com/deploy/apitest/138/843/10171)
    - 基于tf-serving+ideal的服务(推荐)[示例](https://euler.aidigger.com/deploy/apitest/138/843/10170)

## 标准化模型

### 通用配置
- 配置支持.ini格式和.json格式
- 配置支持继承,同名的key会用子配置的值覆盖父配置的值
- [示例配置](examples/tf_base_config.ini)
  
### TextClassify
将一段文本分类,默认支持多标签，每个标签输出带有probability

**输入**
   ```json
{
    "text": "小明的英文名是Jack,小红的英文名叫Rose",
    "label": [{"name":"中性"}, {"name":"正面", "prob":0.7}]
}
```
**输出**：同$input.true_predict
- **TFTextClassifyModel**实现
    - 基于对序列的CLS token做分类实现
    - [示例配置](examples/text_classify/tf_text_classify_config.ini)
    - [示例实验](https://pony.aidigger.com/experiment/7355)

### TextSpanExtract
从文本中抽取文本片段，标记处片段内容、标签、在原文的起止位置

**输入**
```json
{
"text": "小明的英文名是Jack,小红的英文名叫Rose",
"label": [{"text":"小明", "label":"PERSON","span":[0,2]},{"text":"小红","label":"PERSON","span":[12,14]},
          {"text":"Jack","label":"EN","span":[7,11]},{"text":"Rose","label":"EN","span":[19,23]}]
}
```
**输出**:同$input.true_predict
- **TFSeqLabelingModel**实现
    - 基于序列标注+BIO之类的编码方式实现Mention抽取
    - [示例配置](examples/text_span_classify/seq_labeling_config.ini)
    - [示例实验](https://pony.aidigger.com/experiment/update/7337)
    
- **TFGlobalPointerModel**实现
    - 基于对所有可能的span做分类的方式做Mention抽取,基于bert4keras实现，详见https://kexue.fm/archives/8373
    - [示例配置](examples/text_span_classify/global_pointer_config.ini)
    - [示例实验](https://pony.aidigger.com/experiment/7373)

## pony插件
- 所有基于ConfigAI框架的pony插件都在"非标准插件/configAI/"目录下
- ConfigAI的插件还在快速迭代过程中
- 实验细节可以在pony的日志中查看
- 实验结果基于配置文件，在{experiment_dir}/{project}/{model_name}目录下。拥有如下子目录
   - **model/**:最终模型保存的结果，包括模型配置文件，.h5格式的keras模型，tf-serving格式的模型（自动打压缩包）
   - **config.json**:本次实验的配置，可以基于这个配置重跑实验，确保可复现
   - **history.txt**:训练过程中每个epoch的指标输出（需要开启evaluator callback）
   - **eval/**:所有在配置文件"eval_phase_list"中的数据集上的测评结果
   - **output/**:所有在配置文件"output_phase_list"中的数据集上的输出结果
   - **ckpt/**:训练过程中保存的模型文件（需要开启evaluator callback）
- 如果开启tensorboard callback， tensorboard目录在{experiment_dir}/{project}/tensorboard
- 插件列表
    - [TextClassify](https://pony.aidigger.com/plugin/detail/1284/version/0)
    - [TextSpanClassify](https://pony.aidigger.com/plugin/detail/1191/version/0)


## Euler服务
- 所有示例Euler服务在"其他项目/ConfigAIDemo/"目录下
- 所有示例都有ideal/tf-serving+ideal两种架构
- [TextClassify示例服务](https://euler.aidigger.com/project/138/task/849/model)
- [TextSpanClassify示例服务](https://euler.aidigger.com/project/138/task/843/model)

## 工具脚本

## 图形界面
- 目前部署在hpc4上，地址：http://hpc4.yud.io:8501

## notebooks
展示ConfigAI的内部实现，调试时使用
