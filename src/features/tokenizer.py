# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import re

import jieba
import os
from src import config


def load_set(path):
    lines = set()
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                lines.add(line)
    return lines


def tokenizer(line, stopwords_path=config.stopwords_path, cut_type='char'):
    stopwords = load_set(stopwords_path)
    if cut_type == 'word':
        seg_list = jieba.lcut(line)
    elif cut_type == 'char':
        seg_list = list(line)
    else:
        raise ValueError("cut type is word or char.")
    if stopwords:
        seg_list = [x for x in seg_list if x not in stopwords]
    return seg_list


def sent_tokenizer(text, symbol_path=config.sentence_symbol_path):
    symbols = load_set(symbol_path)
    sentence_symbols = "".join(symbols)
    re_s = "[" + sentence_symbols + "]"
    short_sents = re.split(re_s, text)
    return short_sents