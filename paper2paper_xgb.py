import sys
sys.path.append("/home/rqy/.local/lib/python3.5/site-packages")

import time
import json
import pickle
from nltk.text import TextCollection
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
import json
import pickle
import gensim
import random
import math
import multiprocessing
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from get_train_data import compare_two_paper
import json
import pickle
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
# def compare_two_paper(paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract):

def load_nltk_result():
    with open('data/track2/train/train_pub_nltk_result_title.res', 'rb') as r1:
        nltk_title = pickle.load(r1)
    with open('data/track2/train/train_pub_nltk_result_abstract.res', 'rb') as r3:
        nltk_abstract = pickle.load(r3)
    return nltk_title, nltk_abstract


def load_gensim_result():
    with open('data/track2/train/train_pub_gensim_result_title.json', 'rb') as r1:
        gensim_title = pickle.load(r1)

    with open('data/track2/train/train_pub_gensim_result_abstract.json', 'rb') as r3:
        gensum_abstract = pickle.load(r3)
    return gensim_title, gensum_abstract

def f1(negative_example, existing_data_hash_by_name, nltk_title, nltk_abstract, gensim_title, gensum_abstract, model_name,          train_pub, pool_id, is_negative):
    print('pool_id', pool_id, 'begin')
    
    result = {}

    if pool_id == 0:
        x = tqdm(negative_example)
    else:
        x = negative_example

    bst = xgb.Booster(model_file=model_name)
    
    list_ans = []
    if is_negative:
        for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in x:
            cna_x = []
            for paper_id in existing_data_hash_by_name[the_author_name][other_name_author_id]:
                x = compare_two_paper(unass_paper_id, paper_id, train_pub[unass_paper_id], train_pub[paper_id], 
                    author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result=None)
                cna_x.append(x)
            if len(cna_x) == 0:
                continue
            dtest=xgb.DMatrix(np.array(cna_x))
            ypred = bst.predict(dtest)

            for i, paper_id in enumerate(existing_data_hash_by_name[the_author_name][other_name_author_id]):
                result[ (unass_paper_id, paper_id) ] = float(ypred[i])
            list_ans.append( float(ypred[i]) )
    else:
        for unass_author_id, unass_paper_id, the_author_name, author_rank in x:
            cna_x = []
            for paper_id in existing_data_hash_by_name[the_author_name][unass_author_id]:
                x = compare_two_paper(unass_paper_id, paper_id, train_pub[unass_paper_id], train_pub[paper_id], 
                    author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result=None)
                cna_x.append(x)
            if len(cna_x) == 0:
                assert len(existing_data_hash_by_name[the_author_name][unass_author_id]) == 0
                continue
            
            dtest=xgb.DMatrix(np.array(cna_x))
            ypred = bst.predict(dtest)

            for i, paper_id in enumerate(existing_data_hash_by_name[the_author_name][unass_author_id]):
                result[ (unass_paper_id, paper_id) ] = float(ypred[i])
            list_ans.append( float(ypred[i]) )

    print("pool",pool_id , "is finish.", "[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")
    return result, list_ans

def p2p_result(train_pub, model_name, data_result_dir, existing_data_hash_by_name, 
                positive_example, negative_example, nltk_title, nltk_abstract, gensim_title, gensum_abstract):

    #xgb.Booster(model_file=model_name)
    result = {}

    if positive_example is not None:
        print('[positive_example]')
        num_pool = 4
        len_data = len(positive_example)
        print("Length of data", len_data)
        pool = Pool()
        step = len_data//num_pool

        id = 0
        sub_data = []
        jobs = []
        for data in positive_example:
            sub_data.append(data)
            if len(sub_data) >= step:
                jobs.append(pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name,  nltk_title, nltk_abstract, gensim_title, gensum_abstract, model_name, train_pub, id, False)))
                id += 1
                sub_data = []

        if len(sub_data) >= 0:
            jobs.append(pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name,  nltk_title, nltk_abstract, gensim_title, gensum_abstract, model_name, train_pub, id, False)))
            id += 1
            sub_data = []

        pool.close()
        pool.join()

        list_ans = []
        for j in jobs:
            r, a = j.get()
            result.update(r)
            list_ans += a
        
        print('mean ans', np.mean(list_ans))

    print('[negative_example]')

    num_pool = 4
    len_data = len(negative_example)
    print("Length of data", len_data)
    pool = Pool()
    step = len_data//num_pool

    id = 0
    sub_data = []
    jobs = []
    for data in negative_example:
        sub_data.append(data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name,  nltk_title, nltk_abstract, gensim_title, gensum_abstract, model_name, train_pub, id, True)))
            id += 1
            sub_data = []

    if len(sub_data) >= 0:
        jobs.append(pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name,  nltk_title, nltk_abstract, gensim_title, gensum_abstract, model_name, train_pub, id, True)))
        id += 1
        sub_data = []

    pool.close()
    pool.join()

    list_ans = []
    for j in jobs:
        r, a = j.get()
        result.update(r)
        list_ans += a
    
    print('mean ans', np.mean(list_ans))
    

    with open(data_result_dir, 'wb') as file:
        pickle.dump(result, file)


# def compare_two_paper(paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract):

if __name__ == "__main__":
    RETRAIN_P2P_XGB_MODEL = True
    random.seed(0)
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)
    nltk_title, nltk_abstract = load_nltk_result()
    gensim_title, gensum_abstract = load_gensim_result()

    with open('data/track2/train/training_data.pkl', 'rb') as file:
        existing_data_hash_by_name, positive_example, negative_example = pickle.load(file)
    data_result_dir = 'data/track2/train/train_pub_p2p_result_title.res'
    model_name = 'paper2paper_xgb_1.model'


    if RETRAIN_P2P_XGB_MODEL:
        positive_papers = {}
        negative_papers = {}
        id_author_rank = {}
        
        for unass_author_id, unass_paper_id, the_author_name, author_rank in positive_example:
            positive_papers[unass_paper_id] = existing_data_hash_by_name[the_author_name][unass_author_id]
            id_author_rank[unass_paper_id] = author_rank

        for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in negative_example:
            negative_papers[unass_paper_id] = existing_data_hash_by_name[the_author_name][other_name_author_id]
            id_author_rank[unass_paper_id] = author_rank
        
        total = 0

        train_x = []
        train_y = []
        unass_ids = list(positive_papers.keys() & negative_papers.keys())

        for unass_paper_id in tqdm(unass_ids):
            p_list = copy.deepcopy(positive_papers[unass_paper_id])
            n_list = copy.deepcopy(negative_papers[unass_paper_id])
            p_list.remove(unass_paper_id)

            random.shuffle(p_list)
            random.shuffle(n_list)

            for pp,nn in zip(p_list, n_list):
                x = compare_two_paper(unass_paper_id, pp, train_pub[unass_paper_id], train_pub[pp], 
                            id_author_rank[unass_paper_id], nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result=None)

                train_x.append(x)
                train_y.append(1)

                x = compare_two_paper(unass_paper_id, nn, train_pub[unass_paper_id], train_pub[nn], 
                            id_author_rank[unass_paper_id], nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result=None)

                train_x.append(x)
                train_y.append(0)

                total += 2

        print('total', total)
        train_x = np.array(train_x)
        train_y = np.array(train_y)


        
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,test_size = 0.8,random_state = 1)
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

    if True:
        p2p_result(train_pub, model_name, data_result_dir, existing_data_hash_by_name, positive_example, negative_example, nltk_title, nltk_abstract, gensim_title, gensum_abstract)


    