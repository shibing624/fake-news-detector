# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os


class Config:
    def __init__(self):
        self.pwd_path = os.path.abspath(os.path.dirname(__file__))
        # self.model = {
        #     'model1' : AttentionModel,
        #     'model2' : CovlstmModel,
        #     'model3' : DpcnnModel,
        #     'model4' : LstmCovModel,
        #     'model5' : LstmgruModel,
        #     'model6' : TextCnnModel,
        #     'model7' : CapsuleModel,
        #     'model8' : CatboostModel,
        #     # 'model9' : LightGbmModel,
        #     'model10' : XgboostModel
        # }

        self.EMBED_SIZES = 300
        self.MAX_LEN = 1000
        self.BATCH_SIZE = 64
        self.EPOCH = 2

        self.TEXT_X = self.pwd_path + '/../data/News_cut_test_text.txt'
        self.ITEM_TO_ID = '../data/item_to_id_small.pkl'
        self.ID_TO_ITEM = '../data/id_to_item_small.pkl'
        self.EMBEDDING_FILE = "../data/chinese_emb.bin"
        self.TRAIN_X = '../data/All_cut_train_text.txt'
        self.TEST_FILE = '../data/All_cut_test_text.txt'
        self.FEATURES_FILE = '../data/train_features.pkl'
        self.STOP_WORD_FILE = self.pwd_path + '/../data/stopword.txt'
        self.ORIGIN_LABEL_FILE = '../data/News_pic_label_train.txt'
