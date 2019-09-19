'''tfidf-lr stack for education/age/gender'''

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from src import config as cfg
from src.xgboost_lr_model import XGBLR


# -----------------------myfunc-----------------------
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred))
    return np.mean(y_true == y_pred)


# -----------------------load data--------------------
df_all = pd.read_pickle(cfg.data_path + 'all.pkl')
print(df_all.shape)
df_stack = pd.DataFrame(index=range(len(df_all)))
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']
df_train_count = len(df_train)
TR = df_train_count
print('df_train_count:', df_train_count)

X_sp = pickle.load(open(cfg.data_path + 'tfidf_10W_char.feat', 'rb'))

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
    clf = XGBLR(cfg.data_path + 'xgblr_xgb.model')
    clf.train_model(X[tr], y[tr])
    y_pred_va = clf.predict_proba(X[va])
    y_pred_te = clf.predict_proba(X_te)
    print('va acc:', myAcc(y[va], y_pred_va))
    stack[va] += y_pred_va
    stack_te += y_pred_te
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for i in range(stack_all.shape[1]):
    df_stack['tfidf_xgblr_{}_{}'.format('label', i)] = stack_all[:, i]

df_stack.to_csv(cfg.data_path + 'tfidf_xgblr_stack_20W.csv', index=None, encoding='utf8')
print(datetime.now(), 'save tfidf stack done!')

#
#               precision    recall  f1-score   support
#
#            0       0.94      0.92      0.93       381
#            1       0.92      0.94      0.93       388
#
#    micro avg       0.93      0.93      0.93       769
#    macro avg       0.93      0.93      0.93       769
# weighted avg       0.93      0.93      0.93       769
#
# va acc: 0.929778933680104