# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import pickle

import pandas as pd

from src import config
from src.features import ngram
from src.features.count_feature import CountFeatureGenerator
from src.features.tfidf_feature import TfidfFeatureGenerator
from src.features.tokenizer import tokenizer


def process():
    # -----------------------load data--------------------
    data = pd.read_pickle(config.data_file_path)
    print(data.shape)

    print("generate unigram")
    data["text_unigram"] = data["text"].map(lambda x: tokenizer(x))

    print("generate bigram")
    join_str = "_*_"
    data["text_bigram"] = data["text_unigram"].map(lambda x: ngram.getBigram(x, join_str))

    print("generate trigram")
    join_str = "_*_"
    data["text_trigram"] = data["text_unigram"].map(lambda x: ngram.getTrigram(x, join_str))

    with open(config.feature_file_path, 'wb') as f:
        pickle.dump(data, f)
        print('data features saved in ', config.feature_file_path)

    # feature generators
    # svdFG = SvdFeatureGenerator()
    # word2vecFG = Word2VecFeatureGenerator()
    # sentiFG = SentimentFeatureGenerator()
    # generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    generators = [CountFeatureGenerator(), TfidfFeatureGenerator()]

    for g in generators:
        g.process(data[:100])

    for g in generators:
        g.read('train')

    print('done')


if __name__ == "__main__":
    process()
