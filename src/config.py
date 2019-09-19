# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
origin_train_file = pwd_path + '/../data/train.csv'
origin_test_file = pwd_path + '/../data/test_stage1.csv'
stopwords_path = pwd_path + "/../data/stopwords.txt"

output_dir = pwd_path + '/../output/'

data_file_path = output_dir + 'data.pkl'
feature_file_path = output_dir + 'data_feature.pkl'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)