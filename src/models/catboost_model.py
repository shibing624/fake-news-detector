# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from catboost import CatBoostClassifier

from src.models.base_model import BaseClassicModel


class CatBoostModel(BaseClassicModel):
    def __init__(self, num_folds=1, name='catboost'):
        super().__init__(num_folds, name=name)

    def create_model(self):
        model = CatBoostClassifier(loss_function='Logloss', depth=2, n_estimators=10)
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10)
