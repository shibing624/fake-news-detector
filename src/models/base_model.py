# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from src.models.score import compute_acc


class BaseDeepModel(object):
    """
    basic class of all models
    """

    def __init__(self, max_len=400,
                 vocabulary_size=10000,
                 embedding_matrix=None,
                 name='base_deep_model',
                 num_folds=0,
                 batch_size=64,
                 num_classes=2,
                 embedding_dim=128,
                 hidden_dim=128,
                 num_epochs=10):
        self.name = name
        self.batch_size = batch_size
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size
        self.num_folds = 1 if num_folds <= 1 else num_folds
        self.num_epochs = num_epochs
        self.model = self.create_model()

    def create_model(self):
        raise NotImplementedError("need impl create model.")

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError("need impl fit model.")

    def train_predict(self, train_x, train_y, test_x, predict_path):
        """
        train model and predict test result
        :param train_x: list
        :param train_y: list
        :param test_x: list
        :param predict_path: str
        :return:
        """
        print('train_x.shape:', train_x.shape)
        print('train_y.shape:', train_y.shape)
        print('test_x.shape:', test_x.shape)

        n_train = len(train_x)
        train_y = train_y[:n_train]
        print('train_y.shape:', train_y.shape)
        print('n_train', n_train)
        print('y size:', len(train_y), '\nvalue_counts:\n', pd.value_counts(train_y))
        num_class = len(pd.value_counts(train_y))
        print('num_class:', num_class)
        stack = np.zeros((train_x.shape[0], num_class))
        stack_test = np.zeros((test_x.shape[0], num_class))
        scores = []
        if self.num_folds > 1:
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=10)
            for i, (tr, va) in enumerate(kf.split(train_y)):
                print('%s stack:%d/%d' % (str(datetime.now()), i + 1, self.num_folds))
                x_train = train_x[tr]
                y_train = train_y[tr]
                x_valid = train_x[va]
                y_valid = train_y[va]
                self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
                y_pred_valid = self.model.predict_proba(x_valid)
                y_pred_test = self.model.predict_proba(test_x)
                accuracy_rate = compute_acc(y_valid, y_pred_valid)
                print('valid acc:', accuracy_rate)
                scores.append(accuracy_rate)
                stack[va] += y_pred_valid
                stack_test += y_pred_test
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1)
            self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
            y_pred_valid = self.model.predict_proba(x_valid)
            y_pred_test = self.model.predict_proba(test_x)
            y_pred_train = self.model.predict_proba(train_x)
            accuracy_rate = compute_acc(y_valid, y_pred_valid)
            print('valid acc:', accuracy_rate)
            scores.append(accuracy_rate)
            stack += y_pred_train
            stack_test += y_pred_test
        stack_test /= self.num_folds
        score = np.mean(scores)
        print("total scores:", scores, " mean score:", score)
        # output predict result
        data = pd.DataFrame()
        stack_all = np.vstack([stack, stack_test])
        for i in range(stack_all.shape[1]):
            data['{}_{}'.format('label', i)] = stack_all[:, i]
        data.to_csv(predict_path, encoding='utf-8')
        print(datetime.now(), ' predict result save to', predict_path)
        return score


class BaseStaticModel(object):
    def __init__(self, num_folds=0, name='base_static_model', params=None):
        self.params = params
        self.name = name
        self.num_folds = 1 if num_folds <= 1 else num_folds
        self.model = self.create_model()

    def create_model(self):
        raise NotImplementedError("need impl create model.")

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError("need impl fit model.")

    def train_predict(self, train_x, train_y, test_x, predict_path):
        """
        train model and predict test result
        :param train_x: list
        :param train_y: list
        :param test_x: list
        :param predict_path: str
        :return:
        """
        n_train = len(train_x)
        train_y = train_y[:n_train]
        print('n_train', n_train)
        print('y size:', len(train_y), '\nvalue_counts:\n', pd.value_counts(train_y))
        num_class = len(pd.value_counts(train_y))
        print('num_class:', num_class)
        stack = np.zeros((train_x.shape[0], num_class))
        stack_test = np.zeros((test_x.shape[0], num_class))
        scores = []
        if self.num_folds > 1:
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=10)
            for i, (tr, va) in enumerate(kf.split(train_y)):
                print('%s stack:%d/%d' % (str(datetime.now()), i + 1, self.num_folds))
                x_train = train_x[tr]
                y_train = train_y[tr]
                x_valid = train_x[va]
                y_valid = train_y[va]
                self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
                y_pred_valid = self.model.predict_proba(x_valid)
                y_pred_test = self.model.predict_proba(test_x)
                accuracy_rate = compute_acc(y_valid, y_pred_valid)
                print('valid acc:', accuracy_rate)
                scores.append(accuracy_rate)
                stack[va] += y_pred_valid
                stack_test += y_pred_test
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1)
            self.fit_model(self.model, x_train, y_train, x_valid, y_valid)
            y_pred_valid = self.model.predict_proba(x_valid)
            y_pred_test = self.model.predict_proba(test_x)
            y_pred_train = self.model.predict_proba(train_x)
            accuracy_rate = compute_acc(y_valid, y_pred_valid)
            print('valid acc:', accuracy_rate)
            scores.append(accuracy_rate)
            stack += y_pred_train
            stack_test += y_pred_test
        stack_test /= self.num_folds
        score = np.mean(scores)
        print("total scores:", scores, " mean score:", score)
        # output predict result
        data = pd.DataFrame()
        stack_all = np.vstack([stack, stack_test])
        for i in range(stack_all.shape[1]):
            data['{}_{}'.format('label', i)] = stack_all[:, i]
        data.to_csv(predict_path, encoding='utf-8')
        print(datetime.now(), ' predict result save to', predict_path)
        return score
