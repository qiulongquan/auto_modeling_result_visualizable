#!/usr/bin/env python
# -*- coding:utf-8 -*-

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf


# 创建模型架构
def create_model(trial):
    # num of hidden layer
    n_layers = trial.suggest_int('n_layers', 1, 2)
    # dropout_rate
    # dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    layers = []
    for i in range(n_layers):
        layers.append(tf.keras.layers.Dense(256, activation=activation))
    # 由于数据量不多，尽量让数据都去训练这样会提高精度
    # tf.keras.layers.Dropout(dropout_rate)
    layers.append(tf.keras.layers.Dense(128, activation=activation))
    layers.append(tf.keras.layers.Dense(1))
    return tf.keras.Sequential(layers)


# 创建优化器 optimizer
def create_optimizer(trial):
    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    return optimizer


# 创建 trainer
def trainer(trial, x_train, y_train):
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    # さっき作った、ハイパーパラメータを引数に取り、モデルを返す関数
    model = create_model(trial)
    # ハイパーパラメータを引数に取り、最適化手法を返す関数
    optimizer = create_optimizer(trial)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        #                   mae是平均绝对误差
        metrics=['mean_squared_error'])
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=10,
    )
    return model
