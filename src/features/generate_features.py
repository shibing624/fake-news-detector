# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import config
from src.features import ngram
from src.features.count_feature import CountFeatureGenerator
from src.features.sentiment_feature import SentimentFeatureGenerator
from src.features.svd_feature import SvdFeatureGenerator
from src.features.tfidf_feature import TfidfFeatureGenerator
from src.features.tokenizer import tokenizer
from src.features.word2vec_feature import Word2VecFeatureGenerator

# feature generators
generators = [CountFeatureGenerator(),
              TfidfFeatureGenerator(),
              SvdFeatureGenerator(),
              Word2VecFeatureGenerator(),
              SentimentFeatureGenerator()]


def generate_features_label():
    # -----------------------load data--------------------
    if not os.path.exists(config.ngram_feature_path):
        data = pd.read_pickle(config.data_file_path)
        print(data.shape)

        # for test
        # data_a = data[:100]
        # data_b = data[-10:]
        # data = pd.concat([data_a, data_b])
        # print(data.shape)

        print("generate unigram")
        data["text_unigram"] = data["text"].map(lambda x: tokenizer(x))

        print("generate bigram")
        join_str = "_*_"
        data["text_bigram"] = data["text_unigram"].map(lambda x: ngram.getBigram(x, join_str))

        print("generate trigram")
        join_str = "_*_"
        data["text_trigram"] = data["text_unigram"].map(lambda x: ngram.getTrigram(x, join_str))

        with open(config.ngram_feature_path, 'wb') as f:
            pickle.dump(data, f)
            print('data ngram features saved in ', config.ngram_feature_path)
    else:
        data = pd.read_pickle(config.ngram_feature_path)

    for g in generators:
        g.process(data)

    print('done')


def read_features_label():
    data = pd.read_pickle(config.ngram_feature_path)

    # build data
    print("generate feature labels data...")
    data_y = data['label'].values
    train_features = [f for g in generators for f in g.read('train')]
    train_features = [f.toarray() if isinstance(f, csr_matrix) else f for f in train_features]
    for i, f in enumerate(train_features):
        print('shape: ', i, f.shape)
    train_data_x = np.hstack(train_features)
    print('train data_x.shape:', train_data_x.shape)

    test_features = [f for g in generators for f in g.read('test')]
    test_features = [f.toarray() if isinstance(f, csr_matrix) else f for f in test_features]
    test_data_x = np.hstack(test_features)
    print('test data_x.shape:', test_data_x.shape)
    return train_data_x, test_data_x, data_y


if __name__ == "__main__":
    generate_features_label()
