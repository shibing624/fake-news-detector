# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from src import config
from src.features.generate_features import read_features_label
from src.features.onehot_feature import read_onehot_feature_label
from src.models.bert_model import read_bert_feature_label, BertModel
from src.models.catboost_model import CatBoostModel
from src.models.lr_model import LRModel
from src.models.rnn_model import RNNModel
from src.models.textcnn_model import TextCNNModel
from src.models.xgboost_model import XgboostModel


def train_deep_models():
    train_x, test_x, train_y, vocab = read_onehot_feature_label()
    models = [TextCNNModel(max_len=400,
                           num_folds=1,
                           name='textcnn',
                           filter_sizes='3,4,5',
                           embedding_dim=128,
                           hidden_dim=128,
                           num_filters=512,
                           num_classes=2,
                           batch_size=128,
                           vocabulary_size=len(vocab),
                           dropout=0.5,
                           num_epochs=1),
              RNNModel(max_len=400,
                       num_folds=1,
                       name='rnn',
                       embedding_dim=128,
                       hidden_dim=128,
                       num_classes=2,
                       batch_size=128,
                       vocabulary_size=len(vocab),
                       num_epochs=1),
              ]
    for m in models:
        score = m.train_predict(train_x, train_y, test_x, predict_path=config.output_dir + "%s.csv" % m.name)
        print(m.name, score)


def train_bert_model():
    train_x, test_x, train_y = read_bert_feature_label()
    m = BertModel()
    score = m.train_predict(train_x, train_y, test_x, predict_path=config.output_dir + "%s.csv" % m.name)
    print(m.name, score)


def train_classic_models():
    train_x, test_x, train_y = read_features_label()
    models = [LRModel(), XgboostModel(), CatBoostModel()]
    # models = [ CatBoostModel()]
    for m in models:
        score = m.train_predict(train_x, train_y, test_x, predict_path=config.output_dir + "%s.csv" % m.name)
        print(m.name, score)


if __name__ == '__main__':
    # train_deep_models()
    train_bert_model()
    # train_classic_models()

