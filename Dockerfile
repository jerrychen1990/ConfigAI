FROM registry-vpc.cn-hangzhou.aliyuncs.com/eigenlab/config-ai-ideal:tf2.2
RUN pip3 install config_ai -i https://pypi-outer.aidigger.com/simple --trusted-host mirrors.aliyun.com
