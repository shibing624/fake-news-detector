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
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from src import config


# -----------------------myfunc-----------------------
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred))
    return np.mean(y_true == y_pred)


# -----------------------load data--------------------
df_all = pd.read_pickle(config.data_path + 'tr.pkl')
print(df_all.shape)
df_stack = pd.DataFrame(index=range(len(df_all)))
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']
df_train_count = len(df_train)
TR = df_train_count
print('df_train_count:', df_train_count)

X_sp = pickle.load(open(config.data_path + 'tfidf_10W_char.feat', 'rb'))

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
    clf = XGBClassifier()
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

df_stack.to_csv(config.data_path + 'tfidf_lr_stack_20W.csv', index=None, encoding='utf8')
print(datetime.now(), 'save tfidf stack done!')



# char
# 2019-09-18 17:26:15.745512 stack:5/5
#               precision    recall  f1-score   support
#
#            0       0.89      0.94      0.92       381
#            1       0.94      0.89      0.92       388
#
#    micro avg       0.92      0.92      0.92       769
#    macro avg       0.92      0.92      0.92       769
# weighted avg       0.92      0.92      0.92       769
#
# va acc: 0.9167750325097529

# char ngram
# 2019-09-18 17:29:10.312643 stack:5/5
#               precision    recall  f1-score   support
#
#            0       0.90      0.95      0.92       381
#            1       0.95      0.89      0.92       388
#
#    micro avg       0.92      0.92      0.92       769
#    macro avg       0.92      0.92      0.92       769
# weighted avg       0.92      0.92      0.92       769
#
# va acc: 0.9219765929778934
