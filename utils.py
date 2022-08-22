import torch
import torch.nn as nn
from torch import optim

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

#logger.disabled = True  #暫停 logger
#logger.handlers  # logger 內的紀錄程序
#logger.removeHandler  # 移除紀錄程序
#logger.info('xxx', exc_info=True)  # 紀錄堆疊資訊
def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)

    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)

    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    return my_logger

def del_logger(logger):
    # logger.disabled = True  #暫停 logger
    while logger.handlers:  # logger 內的紀錄程序
        h = logger.handlers[0]
        logger.removeHandler(h)

