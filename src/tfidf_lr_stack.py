# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: tfidf-lr stack for education/age/gender
"""
import pickle
from datetime import datetime

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from src import config


# -----------------------myfunc-----------------------
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred))
    return np.mean(y_true == y_pred)


# -----------------------load data--------------------
df_all = pd.read_pickle(config.output_dir + 'all.pkl')
print(df_all.shape)
df_stack = pd.DataFrame(index=range(len(df_all)))
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']

test = df_all[df_all['type'] == 'test']
print(test.head())

df_train_count = len(df_train)
TR = df_train_count
print('df_train_count:', df_train_count)


class Char_Ngram_Tokenizer(object):
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = list(query)
            for gram in [1, 2,3]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.0001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n)
        return tokens


class Char_Tokenizer(object):
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = list(query)
            for gram in [1]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.0001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n)
        return tokens


class Tokenizer(object):
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.0001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n)
        return tokens


vectorizer = TfidfVectorizer(tokenizer=Char_Tokenizer(), sublinear_tf=True)
X_sp = vectorizer.fit_transform(df_all['text'])
print("feature set nums: ", len(vectorizer.vocabulary_))
feature_names = vectorizer.get_feature_names()

ch2_precent = SelectPercentile(chi2, percentile=80)
ch2 = ch2_precent.fit(X_sp[:df_train_count], df_all[:df_train_count]['label'])
X_sp = ch2_precent.transform(X_sp)

features = [feature_names[i] for i in ch2.get_support(indices=True)]
feature_scores = [ch2.scores_[i] for i in ch2.get_support(indices=True)]
sorted_feature = sorted(zip(features, feature_scores), key=lambda x: x[1], reverse=True)
feature_output_file = config.output_dir + 'feature.txt'
with open(feature_output_file, "w", encoding="utf-8") as f_out:
    for id, item in enumerate(sorted_feature):
        f_out.write("\t".join([str(id + 1), item[0], str(item[1])]) + "\n")
print("feature select done,new feature set num: ", len(feature_scores))

pickle.dump(X_sp, open(config.output_dir + 'tfidf_10W_char.feat', 'wb'))

# -----------------------stack for label------------------
TR = df_train_count
n = 5

X = X_sp[:TR]
y = df_all['label'].iloc[:TR]
X_te = X_sp[TR:]
y_te = df_all['label'].iloc[TR:]
num_class = len(pd.value_counts(y))
stack = np.zeros((X.shape[0], num_class))
stack_te = np.zeros((X_te.shape[0], num_class))
kf = KFold(n_splits=n)
for i, (tr, va) in enumerate(kf.split(y)):
    print('%s stack:%d/%d' % (str(datetime.now()), i + 1, n))
    clf = LogisticRegression()
    clf.fit(X[tr], y[tr])
    y_pred_va = clf.predict_proba(X[va])
    y_pred_te = clf.predict_proba(X_te)
    print('va acc:', myAcc(y[va], y_pred_va))
    stack[va] += y_pred_va
    stack_te += y_pred_te
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for i in range(stack_all.shape[1]):
    df_stack['tfidf_lr_{}_{}'.format('label', i)] = stack_all[:, i]

df_stack.to_csv(config.output_dir + 'tfidf_lr_stack_20W.csv', index=None, encoding='utf8')
print(datetime.now(), 'save tfidf stack done!')

# word seg
# 2019-09-18 17:12:26.816302 stack:5/5
#               precision    recall  f1-score   support
#
#            0       0.84      1.00      0.91       381
#            1       1.00      0.81      0.90       388
#
#    micro avg       0.91      0.91      0.91       769
#    macro avg       0.92      0.91      0.90       769
# weighted avg       0.92      0.91      0.90       769
#
# va acc: 0.9050715214564369

# char ngram
# 2019-09-18 17:06:58.260594 stack:5/5
#               precision    recall  f1-score   support
#
#            0       0.87      0.99      0.93       381
#            1       0.99      0.85      0.92       388
#
#    micro avg       0.92      0.92      0.92       769
#    macro avg       0.93      0.92      0.92       769
# weighted avg       0.93      0.92      0.92       769
#
# va acc: 0.9206762028608583

# char
# 2019-09-18 17:14:26.397543 stack:5/5
#               precision    recall  f1-score   support
#
#            0       0.91      0.97      0.94       381
#            1       0.97      0.90      0.94       388
#
#    micro avg       0.94      0.94      0.94       769
#    macro avg       0.94      0.94      0.94       769
# weighted avg       0.94      0.94      0.94       769
#
# va acc: 0.9388816644993498
