import time
import json
import pickle

import numpy as np
import xgboost as xgb

bst = xgb.Booster(model_file='xgb_1.model')
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    score = bst.get_score(importance_type=importance_type)
    print(importance_type)
    for i, key in enumerate(score):
        print(key, score[key])
        if i >= 5:
            break
