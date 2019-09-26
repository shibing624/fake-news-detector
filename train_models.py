# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import pandas as pd

import config
from features.onehot_feature import read_onehot_feature_label
from generate_features import read_features_label


def generate_submit_result(data_path, predict_path, submit_path):
    df = pd.read_pickle(data_path)
    print('df.shape:', df.shape)
    train = df[df['type'] == 'train']
    print('train.shape:', train.shape)
    n_train = train.shape[0]
    print('n_train:', n_train)

    df_predict = pd.read_csv(predict_path)
    df = df.reset_index(drop=True)
    df_predict = df_predict.reset_index(drop=True)
    result_all = pd.concat([df, df_predict], axis=1)
    result = result_all.loc[n_train:, ['id', 'predict_label']]
    result.rename(columns={'predict_label': 'label'}, inplace=True)
    result.to_csv(submit_path, columns=["id", "label"], index=False, encoding='utf-8')
    print('generate submit file:', submit_path)


def train_classic_models():
    from models.catboost_model import CatBoostModel
    from models.lr_model import LRModel
    from models.xgboost_model import XgboostModel  # 该引用必须在bert之前
    train_x, test_x, train_y = read_features_label()
    models = [LRModel(), XgboostModel(), CatBoostModel()]
    for m in models:
        predict_path = config.output_dir + "%s.csv" % m.name
        submit_path = config.output_dir + "%s_submit.csv" % m.name
        score = m.train_predict(train_x, train_y, test_x, predict_path)
        generate_submit_result(config.data_file_path, predict_path, submit_path=submit_path)
        print(m.name, score)


def train_deep_models():
    from models.rnn_model import RNNModel
    from models.textcnn_model import TextCNNModel
    from models.dpcnn_model import DpcnnModel
    train_x, test_x, train_y, vocab = read_onehot_feature_label()
    models = [TextCNNModel(max_len=300,
                           num_folds=1,
                           name='textcnn',
                           filter_sizes='3,4,5',
                           embedding_dim=300,
                           hidden_dim=128,
                           num_filters=512,
                           num_classes=2,
                           batch_size=32,
                           vocabulary_size=len(vocab),
                           dropout=0.5,
                           num_epochs=4),
              RNNModel(max_len=300,
                       num_folds=1,
                       name='rnn',
                       embedding_dim=300,
                       hidden_dim=128,
                       num_classes=2,
                       batch_size=32,
                       vocabulary_size=len(vocab),
                       num_epochs=10),
              DpcnnModel(max_len=300,
                         num_folds=1,
                         name='dpcnn',
                         embedding_dim=300,
                         hidden_dim=256,
                         num_classes=2,
                         batch_size=32,
                         vocabulary_size=len(vocab),
                         num_epochs=10)
              ]
    for m in models:
        predict_path = config.output_dir + "%s.csv" % m.name
        submit_path = config.output_dir + "%s_submit.csv" % m.name
        score = m.train_predict(train_x, train_y, test_x, predict_path)
        generate_submit_result(config.data_file_path, predict_path, submit_path=submit_path)
        print(m.name, score)


def train_bert_model():
    from models.bert_model import read_bert_feature_label, BertModel
    train_x, test_x, train_y = read_bert_feature_label()
    m = BertModel(max_len=200,
                  num_folds=1,
                  name='bert',
                  num_classes=2,
                  batch_size=32,
                  num_epochs=10)
    predict_path = config.output_dir + "%s.csv" % m.name
    # submit_path = config.output_dir + "%s_submit.csv" % m.name
    score = m.train_predict(train_x, train_y, test_x, predict_path)
    # generate_submit_result(config.data_file_path, predict_path, submit_path=submit_path)
    print(m.name, score)


if __name__ == '__main__':
    train_classic_models()
    train_deep_models()
    train_bert_model()
