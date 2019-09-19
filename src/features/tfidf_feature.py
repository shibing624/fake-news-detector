# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


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
        vec_text = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_df=0.8, min_df=2, sublinear_tf=True)
        text_tfidf = vec_text.fit_transform(df['text'].tolist())
        print('text Tfidf.shape:', text_tfidf.shape)
        vocabulary = vec_text.vocabulary_
        print('vocabulary size:%d' % len(vocabulary))
        print('vocabulary list:')
        count = 0
        for k, v in vocabulary.items():
            if count < 10:
                print("%s\t%s" % (k, v))
                count += 1

        # save train and test into separate files
        tfidf_train = text_tfidf[:n_train, :]
        tfidf_train_feature_path = config.output_dir + "train.text.tfidf.pkl"
        with open(tfidf_train_feature_path, "wb") as f:
            pickle.dump(tfidf_train, f)
        print('text tfidf features of training set saved in %s' % tfidf_train_feature_path)

        if n_test > 0:
            # test set is available
            tfidf_test = text_tfidf[n_train:, :]
            tfidf_test_feature_path = config.output_dir + "test.text.tfidf.pkl"
            with open(tfidf_test_feature_path, "wb") as f:
                pickle.dump(tfidf_test, f)
            print('text tfidf features of test set saved in %s' % tfidf_test_feature_path)

            # 3). compute cosine similarity between TEXT tfidf features and DEBUNKING features
            # sim_tfidf = np.asarray(map(cosine_sim, text_tfidf, text_tfidf))[:, np.newaxis]
            # print('sim_tfidf.shape:', sim_tfidf.shape)
            # sim_tfidf_train = sim_tfidf[:n_train]
            # sim_tfidf_train_path = config.output_dir + "train.sim.tfidf.pkl"
            # with open(sim_tfidf_train_path, "wb") as f:
            #     pickle.dump(sim_tfidf_train, f)
            # print('tfidf sim. features of training set saved in %s' % sim_tfidf_train_path)
            #
            # if n_test > 0:
            #     # test set is available
            #     sim_tfidf_test = sim_tfidf[n_train:]
            #     sim_tfidf_test_path = config.output_dir + "test.sim.tfidf.pkl"
            #     with open(sim_tfidf_test_path, "wb") as f:
            #         pickle.dump(sim_tfidf_test, f)
            #     print('tfidf sim. features of test set saved in %s' % sim_tfidf_train_path)

    def read(self, header='train'):
        text_feature_path = config.output_dir + "%s.text.tfidf.pkl" % header
        with open(text_feature_path, "rb") as f:
            text_tfidf = pickle.load(f)

        # sim_feature_path = config.output_dir + "%s.sim.tfidf.pkl" % header
        # with open(sim_feature_path, "rb") as f:
        #     sim_tfidf = pickle.load(f)

        print('text_tfidf.shape:', text_tfidf.shape)
        # print('sim_tfidf.shape:', sim_tfidf.shape)

        # return [text_tfidf, sim_tfidf.reshape(-1, 1)]
        return [text_tfidf]