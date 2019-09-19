# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 1.concat train data and test data
   2.use lr to fill null label
"""

import pandas as pd

from src import config

label_dict = {'0': 0,  # true
              '1': 1  # fake
              }

# ----------------------load data--------------------------------

df_tr = pd.read_csv(config.origin_train_file)
df_tr['type'] = 'train'

df_te = pd.read_csv(config.origin_test_file)
df_te['label'] = 0
df_te['type'] = 'test'

print(df_tr.shape)
print(df_te.shape)

print(df_tr.head(2))
print(df_tr.tail(2))

print(df_te.head(2))

print(df_tr['label'].value_counts())

df_all = pd.concat([df_tr, df_te]).fillna(0)
df_all.to_pickle(config.data_file_path, protocol=2)

# train count: 38471
# test count:  4000
