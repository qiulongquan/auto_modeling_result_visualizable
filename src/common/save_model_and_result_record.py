#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import os
from src.basic_config import NN_model_list


def save_model_object(model, sub_path, model_category, model_name):
    # 创建文件目录
    base_dir = 'model_and_record'
    path = os.path.join(base_dir, sub_path, model_category)
    if not os.path.exists(path):
        os.makedirs(path)

    # 神经模型保存
    if model_name in NN_model_list:
        model_name = model_name
        path = os.path.join(path, model_name)
        model.save(path)
    else:
        # 普通传统模型保存
        model_name = model_name + '.pkl'
        path = os.path.join(path, model_name)
        with open(path, mode='wb') as fp:
            pickle.dump(model, fp)


def load_model_object(sub_path, model_category, model_name):
    base_dir = 'model_and_record'
    path = os.path.join(base_dir, sub_path, model_category, model_name)
    with open(path, "rb") as f:
        model = pickle.load(f)  # 読み出し
    return model


def save_record_object(result_record_df, sub_path, file_name):
    # 创建文件目录
    base_dir = 'model_and_record'
    path = os.path.join(base_dir, sub_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # 保存记录
    file_name = file_name + '.pkl'
    path = os.path.join(path, file_name)
    with open(path, mode='wb') as fp:
        pickle.dump(result_record_df, fp)


def load_record_object(sub_path, file_name):
    base_dir = 'model_and_record'
    path = os.path.join(base_dir, sub_path, file_name)
    with open(path, "rb") as f:
        result_record_df = pickle.load(f)  # 読み出し
    return result_record_df
