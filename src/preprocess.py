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
print(df_tr.shape)
# 文本去重
df_tr = df_tr.drop_duplicates(["text"], keep="first")
print('drop_duplicates:', df_tr.shape)

# 文本去太长的, 删除681个长的
df_tr = df_tr[df_tr['text'].str.len() < 300]
print('del long text:', df_tr.shape)

df_te = pd.read_csv(config.origin_test_file)
df_te['label'] = 0
df_te['type'] = 'test'

print(df_te.shape)

print(df_tr.head(2))
print(df_tr.tail(2))

print(df_te.head(2))

print(df_tr['label'].value_counts())

df_all = pd.concat([df_tr, df_te])

df_all.to_pickle(config.data_file_path)

# train count: 38471， del duplicates count: 33414
# test count:  4000
