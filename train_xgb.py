import json
import pickle

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


with open('data/track2/train/train_x.pkl', 'rb') as rb:
    train_x = pickle.load(rb)
with open('data/track2/train/train_y.pkl', 'rb') as rb:
    train_y = pickle.load(rb)

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,test_size = 0.2,random_state = 1)
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)

params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':4,
        'lambda':10,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2,
        'eta': 0.025,
        'seed':0}

watchlist = [(dtrain, 'train')]
bst=xgb.train(params, dtrain, num_boost_round=200, evals=watchlist)
#输出概率
ypred=bst.predict(dtest)
 
# 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
y_pred = (ypred >= 0.3)*1
 
from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print(metrics.confusion_matrix(test_y,y_pred))

print('----- save_model -----')
bst.save_model('xgb_1.model')
tar = xgb.Booster(model_file='xgb_1.model')

ypred=tar.predict(dtest)
 
# 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
y_pred = (ypred >= 0.3)*1
 
from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print(metrics.confusion_matrix(test_y,y_pred))