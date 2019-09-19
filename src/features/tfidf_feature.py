# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.math_util import cosine_sim


class TfidfFeatureGenerator(object):
    def __init__(self, name='tfidfFeatureGenerator'):
        self.name = name

    def process(self, df):
        # 1). create strings based on text
        train = df[df['type'] == 'train']
        print(train.head())

        test = df[df['type'] == 'test']
        print(test.head())
        print('train.shape:', train.shape)

        n_train = train.shape[0]
        print('n_train:', n_train)
        n_test = test.shape[0]
        print('n_test:', n_test)

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["text_unigram"])  # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        text_tfidf = vecH.fit_transform(df['text_unigram'].map(lambda x: ' '.join(x)))
        print('text Tfidf.shape:', text_tfidf.shape)

        # save train and test into separate files
        tfidf_train = text_tfidf[:n_train, :]
        tfidf_train_feature_path = "train.text.tfidf.pkl"
        with open(tfidf_train_feature_path, "wb") as f:
            pickle.dump(tfidf_train, f, -1)
        print('text tfidf features of training set saved in %s' % tfidf_train_feature_path)

        if n_test > 0:
            # test set is available
            tfidf_test = text_tfidf[n_train:, :]
            tfidf_test_feature_path = "test.text.tfidf.pkl"
            with open(tfidf_test_feature_path, "wb") as f:
                pickle.dump(tfidf_test, f)
            print('headline tfidf features of test set saved in %s' % tfidf_test_feature_path)

        # 4). compute cosine similarity between TEXT tfidf features and DEBUNKING features
        sim_tfidf = np.asarray(map(cosine_sim, text_tfidf, text_tfidf))[:, np.newaxis]
        print('simTfidf.shape:', sim_tfidf.shape)
        sim_tfidf_train = sim_tfidf[:n_train]
        sim_tfidf_train_path = "train.sim.tfidf.pkl"
        with open(sim_tfidf_train_path, "wb") as f:
            pickle.dump(sim_tfidf_train, f)
        print('tfidf sim. features of training set saved in %s' % sim_tfidf_train_path)

        if n_test > 0:
            # test set is available
            sim_tfidf_test = sim_tfidf[n_train:]
            sim_tfidf_test_path = "test.sim.tfidf.pkl"
            with open(sim_tfidf_test_path, "wb") as f:
                pickle.dump(sim_tfidf_test, f)
            print('tfidf sim. features of test set saved in %s' % sim_tfidf_train_path)

    def read(self, header='train'):
        text_feature_path = "%s.text.tfidf.pkl" % header
        with open(text_feature_path, "rb") as f:
            text_tfidf = pickle.load(f)

        filename_simtfidf = "%s.sim.tfidf.pkl" % header
        with open(filename_simtfidf, "rb") as f:
            sim_tfidf = pickle.load(f)

        print('text_Tfidf.shape:', text_tfidf.shape)
        print('sim_tfidf.shape:', sim_tfidf.shape)
        return [text_tfidf, sim_tfidf.reshape(-1, 1)]
