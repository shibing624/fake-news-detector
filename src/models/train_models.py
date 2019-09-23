# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from src import config
from src.features.generate_features import read_features_label, read_onehot_feature_label
from src.models.lr_model import LRModel
from src.models.textcnn_model import TextCNNModel
from src.models.xgboost_model import XgboostModel


def train_deep_models():
    train_x, test_x, train_y = read_onehot_feature_label()
    m = TextCNNModel()
    score = m.train_predict(train_x, train_y, test_x, predict_path=config.output_dir + "%s.csv" % m.name)
    print(m.name, score)


def train_classic_models():
    train_x, test_x, train_y = read_features_label()
    models = [LRModel(), XgboostModel()]
    for m in models:
        score = m.train_predict(train_x, train_y, test_x, predict_path=config.output_dir + "%s.csv" % m.name)
        print(m.name, score)


if __name__ == '__main__':
    train_classic_models()
    train_deep_models()
