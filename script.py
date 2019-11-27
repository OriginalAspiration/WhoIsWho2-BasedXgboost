import sys
sys.path.append("/home/rqy/.local/lib/python3.5/site-packages")

import time
import json
import pickle
import nltk
from nltk.text import TextCollection
import gensim
from gensim import corpora, models, similarities
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from format_process import multi_process_format_data
import train_model
from count_name import replace_str
from get_train_data import compare_paper_with_set
import multiprocessing
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import paper2paper_xgb 
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from train_nn import Net


def load_nltk_result():
    with open('data/track2/test/test_pub_nltk_result_title.res', 'rb') as r1:
        nltk_title = pickle.load(r1)
    with open('data/track2/test/test_pub_nltk_result_abstract.res', 'rb') as r3:
        nltk_abstract = pickle.load(r3)
    return nltk_title, nltk_abstract


def load_gensim_result():
    with open('data/track2/test/test_pub_gensim_result_title.json', 'rb') as r1:
        gensim_title = pickle.load(r1)
    with open('data/track2/test/test_pub_gensim_result_abstract.json', 'rb') as r3:
        gensum_abstract = pickle.load(r3)
    return gensim_title, gensum_abstract

def load_p2p_result():
    with open('data/track2/test/test_pub_p2p_result_title.res', 'rb') as r1:
        p2p_result = pickle.load(r1)
    return p2p_result

def get_model_func(model_name, model_type='xgb'):
    def xgb_model_func(array_x):
        bst = xgb.Booster(model_file=model_name)
        dtest=xgb.DMatrix(array_x)
        return bst.predict(dtest)

    def pytorch_model_func(array_x):
        tensor_x = torch.tensor(array_x, dtype=torch.float32)
        return net(tensor_x).squeeze().detach().numpy()

    if model_type == 'xgb':
        return xgb_model_func
    elif model_type == 'pytorch':
        net = torch.load(model_name)
        return pytorch_model_func
    else:
        assert False, '--- Unknown model'


def f(cna_valid_unass_competition, cna_valid_pub, test_alter_pub, kdd_data=None, kdd_data_triplet=None, pool_id=0):
    global model_call_func_1
    global model_call_func_2
    print('pool_id', pool_id, 'begin')
    result_dict = {}
    error_times = 0
    if pool_id == 0:
        x = tqdm(cna_valid_unass_competition)
    else:
        x = cna_valid_unass_competition
    for unass_data in x:
        unass_paper_id = unass_data[:8]
        author_rank = int(unass_data[9:])
        unass_paper_info = cna_valid_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank]['name']

        cna_x = []
        id_list = []
        try:
            for same_name_author_id in whole_data_hash_by_name[replace_str(the_author_name)]:
                x = compare_paper_with_set(whole_data_hash_by_name[replace_str(the_author_name)][same_name_author_id], 
                                        unass_paper_id, test_alter_pub, author_rank, nltk_title, nltk_abstract, gensim_title,
                                        gensum_abstract, p2p_result, kdd_data, kdd_data_triplet)
                cna_x.append(x)
                id_list.append(same_name_author_id)

            ypred_1 = model_call_func_1(np.array(cna_x))
            ypred_2 = model_call_func_2(np.array(cna_x))
            ypred = ypred_1 + ypred_2
            predicted_author_id = id_list[np.argsort(ypred)[-1].item()]
            if predicted_author_id not in result_dict:
                result_dict[predicted_author_id] = []
            result_dict[predicted_author_id].append(unass_paper_id)
        except Exception as e:
            error_times += 1
            print('error_times: ', error_times)
            print(e)
    print("pool",pool_id , "is finish.", "[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")
    return result_dict

if __name__ == "__main__":
    UPDATE_KDD_BY_SELF = True
    INIT_TEST_ALTER_PUB = True
    INIT_P2P_XGB = True
    TRAIN_MODEL = True

    model_call_func_1 = get_model_func('xgb_1.model', 'xgb')
    model_call_func_2 = get_model_func('nn_1.model', 'pytorch')

    with open('data/track2/cna_data/cna_valid_unass_competition.json', 'r') as r:
        cna_valid_unass_competition = json.load(r)
    with open('data/track2/cna_data/cna_valid_pub.json', 'r') as r:
        cna_valid_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile_pub.json', 'r') as r:
        whole_author_profile_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile.json', 'r') as r:
        whole_author_profile = json.load(r)
    #kdd data
    if UPDATE_KDD_BY_SELF:
        with open('data/kdd_embedding/whole_pid_order_to_features.pkl', 'rb') as rb:
            kdd_data = pickle.load(rb)
        with open('data/kdd_embedding/whole_pid_order_to_features_triplet.pkl', 'rb') as rb:
            kdd_data_triplet = pickle.load(rb)
        with open('data/kdd_embedding/cna_pid_order_to_features_use_whole.pkl', 'rb') as rb:
            kdd_data_cna = pickle.load(rb)
        with open('data/kdd_embedding/cna_pid_order_to_features_triplet_use_whole.pkl', 'rb') as rb:
            kdd_data_triplet_cna = pickle.load(rb)
        kdd_data.update(kdd_data_cna)
        kdd_data_triplet.update(kdd_data_triplet_cna)
        with open('data/kdd_embedding/kdd_data_script.pkl', 'wb') as wb:
            pickle.dump(kdd_data, wb)
        with open('data/kdd_embedding/kdd_data_triplet_script.pkl', 'wb') as wb:
            pickle.dump(kdd_data_triplet, wb)
    else:
        with open('data/kdd_embedding/kdd_data_script.pkl', 'rb') as rb:
            kdd_data = pickle.load(rb)
        with open('data/kdd_embedding/kdd_data_triplet_script.pkl', 'rb') as rb:
            kdd_data_triplet = pickle.load(rb)
    print('--- script.py load data finish ---')

    whole_data_hash_by_name = {}
    for person_id in whole_author_profile:
        real_name = whole_author_profile[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in whole_data_hash_by_name:
            whole_data_hash_by_name[replaced_real_name] = {}
        whole_data_hash_by_name[replaced_real_name][person_id] = whole_author_profile[person_id]['papers']
    
    # format_process
    test_pub = {}
    negative_example = []
    file_name_alter_pub = 'data/track2/test/test_pub_alter.json'
    
    for index, unass_data in tqdm(enumerate(cna_valid_unass_competition)):
        unass_paper_id = unass_data[:8]
        author_rank = int(unass_data[9:])
        unass_paper_info = cna_valid_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank]['name']
        old_name = the_author_name
        the_author_name = replace_str(the_author_name)

        test_pub[unass_paper_id] = unass_paper_info
        #if unass_paper_id == "Q8tZ3xgK":
        #    print('find Q8tZ3xgK')
        if the_author_name in whole_data_hash_by_name:
            for same_name_author_id in whole_data_hash_by_name[the_author_name]:
                negative_example.append([None, unass_paper_id, the_author_name, author_rank, same_name_author_id])
                for paper_id in whole_data_hash_by_name[the_author_name][same_name_author_id]:
                    test_pub[paper_id] = whole_author_profile_pub[paper_id]
        else:
            #print('the_author_name', the_author_name)
            #print('old_name', old_name)
            pass
    if INIT_TEST_ALTER_PUB:
        print('----- INIT_TEST_ALTER_PUB START -----')
        # format_process.save_format_data(test_pub, file_name_alter_pub)
        test_alter_pub = multi_process_format_data(test_pub)
        with open(file_name_alter_pub, 'w', encoding='utf-8') as w:
            w.write(json.dumps(test_alter_pub))
    else:
        with open(file_name_alter_pub, 'r') as r:
            test_alter_pub = json.load(r)

    #assert False

    #train_model
    if TRAIN_MODEL:        
        print('----- TRAIN_MODEL START -----')
        if True:
            data_model_dir = 'data/track2/test/test_pub_nltk_' + 'model_'
            data_result_dir = 'data/track2/test/test_pub_nltk_' + 'result_'
            train_model.nltk_idf(test_alter_pub, data_model_dir)
            train_model.nltk_tf(test_alter_pub, data_model_dir, data_result_dir, 
                                whole_data_hash_by_name, None, negative_example)

        if True:
            data_model_dir = 'data/track2/train/train_pub_gensim_' + 'model_'
            data_result_dir = 'data/track2/test/test_pub_gensim_' + 'result_'
            train_model.gensim_result(test_pub, data_model_dir, data_result_dir,
                                whole_data_hash_by_name, None, negative_example)
    
    nltk_title, nltk_abstract = load_nltk_result()
    gensim_title, gensum_abstract = load_gensim_result()

    if INIT_P2P_XGB:
        print('----- INIT_P2P_XGB START -----')
        #p2p_result(train_pub, model_namem, data_result_dir, existing_data_hash_by_name,  positive_example, negative_example, nltk_title, nltk_abstract, gensim_title, gensum_abstract):
        data_result_dir = 'data/track2/test/test_pub_p2p_result_title.res'
        paper2paper_xgb.p2p_result(test_pub, 'paper2paper_xgb_1.model', data_result_dir, whole_data_hash_by_name, None, negative_example, nltk_title, nltk_abstract, gensim_title, gensum_abstract)
    
    p2p_result = load_p2p_result()
    # p2p_result = None
    #gensim_title, gensum_abstract = None, None

    #cna_valid_unass_competition = cna_valid_unass_competition[:20]
    num_pool = 4
    len_data = len(cna_valid_unass_competition)
    print("Length of data", len_data)
    print('----- mutiprocess run f START -----')
    pool = Pool()#train_pub
    step = len_data//num_pool
    id = 0
    sub_data = []
    jobs = []
    for one_data in cna_valid_unass_competition:
        sub_data.append(one_data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f, args=(sub_data, cna_valid_pub, test_alter_pub, kdd_data, kdd_data_triplet, id)))
            id += 1
            sub_data = []

    if len(sub_data) > 0:
        jobs.append(pool.apply_async(f, args=(sub_data, cna_valid_pub, test_alter_pub, kdd_data, kdd_data_triplet, id)))
        id += 1
        sub_data = {}
    
    pool.close()
    pool.join()

    results = {}
    for j in jobs:
        sub_results = j.get()
        print( len(sub_results) )
        for y in sub_results:
            if y in results:
                results[y] += sub_results[y]
            else:
                results[y] = sub_results[y]
    print( len(results) )

    with open('result.json', 'w') as w:
        w.write(json.dumps(results))

