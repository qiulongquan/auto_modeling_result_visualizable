#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pathlib import Path
import logging
import time
import os

# logger disabled定義
logger_disabled = False
# loggingファイル名定義
logging_file_name = time.strftime('%Y%m%d_%H%M%S', time.localtime(
    time.time())) + '.log'
# logging_path定義
log_base_path = 'log'
# logging name定義
logging_name = 'model_process_log'
# logging 出力レベル定義
level_info = logging.INFO
# logging フォーマット定義
formatting = '%(asctime)s [%(levelname)s] %(message)s'

app_home = str(Path(__file__).parents[2])
# logging设定
logger = logging.getLogger(logging_name)
logger.setLevel(level=level_info)
filehandle = logging.FileHandler(os.path.join(app_home, log_base_path,
                                              logging_file_name),
                                 encoding="utf-8")
filehandle.setLevel(level=level_info)
formatter = logging.Formatter(formatting)
filehandle.setFormatter(formatter)
logger.addHandler(filehandle)


def log_output(message):
    logger.info(message)
