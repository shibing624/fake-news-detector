# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import kashgari
import pandas as pd
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BiLSTM_Model

from src import config
from src.models.base_model import BaseDeepModel
from src.models.tokenization import BasicTokenizer


def read_bert_feature_label():
    tokenizer = BasicTokenizer()

    df = pd.read_pickle(config.ngram_feature_path)
    train = df[df['type'] == 'train']
    print(train.head())

    test = df[df['type'] == 'test']
    print(test.head())
    print('train.shape:', train.shape)
    print('test.shape:', test.shape)

    n_train = train.shape[0]
    print('n_train:', n_train)
    n_test = test.shape[0]
    print('n_test:', n_test)

    df['bert_tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
    train_x = df['bert_tokens'][:n_train].tolist()
    test_x = df['bert_tokens'][n_train:].tolist()
    train_y = df['label'][:n_train].tolist()

    print('len train_x:', len(train_x))
    print('len test_x:', len(test_x))
    print('len train_y:', len(train_y))
    return train_x, test_x, train_y


class BertModel(BaseDeepModel):
    def __init__(self, bert_path=config.pretrained_bert_path,
                 max_len=400,
                 num_folds=1,
                 name='bert',
                 embedding_dim=128,
                 hidden_dim=128,
                 num_classes=2,
                 batch_size=64,
                 vocabulary_size=20000,
                 num_epochs=1,
                 model_path=config.output_dir + 'bert.model'):
        self.bert_path = bert_path
        self.model_path = model_path
        super(BertModel, self).__init__(max_len=max_len,
                                        num_folds=num_folds,
                                        name=name,
                                        num_classes=num_classes,
                                        vocabulary_size=vocabulary_size,
                                        embedding_dim=embedding_dim,
                                        hidden_dim=hidden_dim,
                                        batch_size=batch_size,
                                        num_epochs=num_epochs)

    def create_model(self):
        print("Creating bert embedding bi-lstm Model...")

        # 初始化 Embedding
        embed = BERTEmbedding(model_folder=self.bert_path,
                              task=kashgari.CLASSIFICATION,
                              sequence_length=self.max_len)

        # 使用 embedding 初始化模型
        model = BiLSTM_Model(embed)
        print(BiLSTM_Model.get_default_hyper_parameters())
        return model

    def fit_model(self, model, x_train, y_train, x_valid, y_valid):
        model.fit(x_train, y_train, x_valid, y_valid,
                  batch_size=self.batch_size, epochs=self.num_epochs)
