# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import jieba

from src import config


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def tokenizer(line, stopwords_path=config.stopwords_path, cut_type='char'):
    stopwords = read_stopwords(stopwords_path)
    if cut_type == 'word':
        seg_list = jieba.lcut(line)
    elif cut_type == 'char':
        seg_list = list(line)
    else:
        seg_list = list(line)
    if stopwords:
        seg_list = [x for x in seg_list if x not in stopwords]
    return seg_list
