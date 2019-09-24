# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from sklearn.linear_model import LogisticRegression

from models.base_model import BaseClassicModel


class LRModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='lr'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = LogisticRegression()
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        print('x_train.shape:', x_train.shape)
        print('y_train.shape:', y_train.shape)
        print('x_valid.shape:', x_valid.shape)
        print('y_valid.shape:', y_valid.shape)
        model.fit(x_train, y_train)
