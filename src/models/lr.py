# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src import config
from src.score import compute_acc


def load_feature_label(path):
    with open(path, 'rb') as f:
        [train_data_x, test_data_x, data_y] = pickle.load(f)
    return train_data_x, test_data_x, data_y


def cv(feature_label_path, data_file_path, n=5):
    train_data_x, test_data_x, data_y = load_feature_label(feature_label_path)
    n_train = len(train_data_x)
    y = data_y[:n_train]
    data = pd.read_pickle(data_file_path)
    print(data.shape)

    # -----------------------stack for label------------------
    num_class = len(pd.value_counts(y))
    stack = np.zeros((train_data_x.shape[0], num_class))
    stack_test = np.zeros((test_data_x.shape[0], num_class))
    kf = KFold(n_splits=n)
    for i, (tr, va) in enumerate(kf.split(y)):
        print('%s stack:%d/%d' % (str(datetime.now()), i + 1, n))
        x_train = train_data_x[tr]
        y_train = y[tr]
        x_valid = train_data_x[va]
        y_valid = y[va]
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_pred_valid = clf.predict_proba(x_valid)
        y_pred_test = clf.predict_proba(test_data_x)
        print('valid acc:', compute_acc(y_valid, y_pred_valid))
        stack[va] += y_pred_valid
        stack_test += y_pred_test
    stack_test /= n
    stack_all = np.vstack([stack, stack_test])
    for i in range(stack_all.shape[1]):
        data['lr_{}_{}'.format('label', i)] = stack_all[:, i]

    data.to_csv(config.output_dir + 'lr_stack.csv', index=None, encoding='utf-8')
    print(datetime.now(), 'save lr stack done!')


if __name__ == '__main__':
    cv(config.features_label_path, config.ngram_feature_path)