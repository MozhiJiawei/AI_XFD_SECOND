import logging
import logging.config
import os
import json


def setup_logging(default_path="logging.json", default_level=logging.INFO, env_key="LOG_CFG"):
    path = default_path

    # 尝试从环境变量读取日志配置
    value = os.getenv(env_key, None)
    if value:
        path = value
        print("从环境变量中获取到配置地址:", value)
    else:
        print("没有从环境变量中获取到配置地址")

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    # 从json文件中加载日志配置
    if os.path.exists(path):
        with open(path, "r") as f:

            # 读取json文件为字典
            config = json.load(f)
            # 使用字典进行全局日志配置
            logging.config.dictConfig(config)
    else:

        # 没有读到配置文件时胡乱配置一个
        logging.basicConfig(level=default_level)
