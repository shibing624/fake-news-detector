# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import pandas as pd
from src.models.tokenization import BasicTokenizer
from src import config
tokenizer = BasicTokenizer()

df = pd.read_pickle(config.ngram_feature_path)
# 进行分词处理
df = df[df['type'] == 'train']
df['cutted'] = df['text'].apply(lambda x: tokenizer.tokenize(x))

# 准备训练测试数据集
train_x = list(df['cutted'][:int(len(df)*0.7)])
train_y = list(df['label'][:int(len(df)*0.7)])

valid_x = list(df['cutted'][int(len(df)*0.7):int(len(df)*0.85)])
valid_y = list(df['label'][int(len(df)*0.7):int(len(df)*0.85)])

test_x = list(df['cutted'][int(len(df)*0.85):])
test_y = list(df['label'][int(len(df)*0.85):])

print('len train_y:', len(train_y))
print('len valid_y:', len(valid_y))
print('len test_y:', len(test_y))
import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BiLSTM_Model

BERT_PATH = '/Users/xuming06/Codes/bert/data/chinese_L-12_H-768_A-12'


# 初始化 Embedding
embed = BERTEmbedding(BERT_PATH,
                     task=kashgari.CLASSIFICATION,
                     sequence_length=64)

# 使用 embedding 初始化模型
model = BiLSTM_Model(embed)
# 先只训练一轮
model.fit(train_x, train_y, valid_x, valid_y, batch_size=32, epochs=1)

model.evaluate(test_x, test_y, batch_size=32)
