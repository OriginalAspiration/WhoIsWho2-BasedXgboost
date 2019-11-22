import json
import pickle

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

model_name = 'xgb_1.model'
model2_name = 'gdbt_1.model'
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
        'seed':0,}
        #'scale_pos_weight' : 100}

watchlist = [(dtrain, 'train')]
bst=xgb.train(params, dtrain, num_boost_round=200, evals=watchlist)
#输出概率
ypred=bst.predict(dtest)
y_pred1 = ypred
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
bst.save_model(model_name)
#tar = xgb.Booster(model_file=model_name)



import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

est = GradientBoostingRegressor(
loss='ls',      ##默认ls损失函数'ls'是指最小二乘回归lad'（最小绝对偏差）'huber'是两者的组合
n_estimators=100, ##默认100 回归树个数 弱学习器个数
learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
max_depth=3,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
subsample=1,  ##用于拟合个别基础学习器的样本分数 选择子样本<1.0导致方差的减少和偏差的增加
min_samples_split=2, ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
min_samples_leaf=1, ##叶节点所需的最小样本数  如果是浮点数代表是百分比
max_features=None, ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
max_leaf_nodes=None, ##叶节点的数量 None不限数量
min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
verbose=1,  ##打印输出 大于1打印每棵树的进度和性能
warm_start=False, ##True在前面基础上增量训练 False默认擦除重新训练 增加树
random_state=0  ##随机种子-方便重现
).fit(train_x, train_y)


#mean_squared_error(test_y, est.predict(test_x))

joblib.dump(est, filename=model2_name)
est = joblib.load(model2_name)

y_pred = est.predict(test_x)
y_pred2 = y_pred
y_pred = (y_pred >= 0.3)*1

print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print(metrics.confusion_matrix(test_y,y_pred))



y_pred = (y_pred1+y_pred2)/2
print(y_pred1[:20],y_pred2[:20],y_pred[:20],test_y[:20])
y_pred = (y_pred >= 0.3)*1



print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print(metrics.confusion_matrix(test_y,y_pred))