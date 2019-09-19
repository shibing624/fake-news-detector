# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import pickle

import pandas as pd

from src import config
from src.features.sentiment import sentiment_classify


class SentimentFeatureGenerator(object):
    def __init__(self, name='sentimentFeatureGenerator'):
        self.name = name

    def process(self, df):
        print('generating sentiment features')

        train = df[df['type'] == 'train']
        test = df[df['type'] == 'test']

        n_train = train.shape[0]
        print('n_train:', n_train)
        n_test = test.shape[0]
        print('n_test:', n_test)

        # calculate the polarity score of each sentence
        def compute_sentiment(text):
            """
            "sentiment":2,    //表示情感极性分类结果, 0:负向，1:中性，2:正向
            "confidence":0.40, //表示分类的置信度
            "positive_prob":0.73, //表示属于积极类别的概率
            "negative_prob":0.27  //表示属于消极类别的概率
            :param text:
            :return:
            """
            return pd.DataFrame([sentiment_classify(text)]).mean()

        # df['text_sents'] = df['text'].apply(lambda x: sent_tokenizer(x))
        df = pd.concat([df, df['text'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'sentiment': 'text_sentiment', 'confidence': 'text_confidence',
                           'positive_prob': 'text_positive_prob', 'negative_prob': 'text_negative_prob'}, inplace=True)
        text_sentiment = df[['text_sentiment', 'text_confidence', 'text_positive_prob', 'text_negative_prob']].values
        print('text sentiment.shape:', text_sentiment.shape)

        text_sentiment_train = text_sentiment[:n_train, :]
        text_sentiment_train_path = config.output_dir + "train.text.sentiment.pkl"
        with open(text_sentiment_train_path, "wb") as f:
            pickle.dump(text_sentiment_train, f)
        print('text sentiment features of training set saved in %s' % text_sentiment_train_path)

        if n_test > 0:
            # test set is available
            text_sentiment_test = text_sentiment[n_train:, :]
            text_sentiment_test_path = config.output_dir + "test.text.sentiment.pkl"
            with open(text_sentiment_test_path, "wb") as f:
                pickle.dump(text_sentiment_test, f)
            print('text sentiment features of test set saved in %s' % text_sentiment_test_path)
        print('text sentiment done')

    def read(self, header='train'):
        text_sentiment_path = config.output_dir + "%s.text.sentiment.pkl" % header
        with open(text_sentiment_path, "rb") as f:
            text_sentiment = pickle.load(f)
        return [text_sentiment]
