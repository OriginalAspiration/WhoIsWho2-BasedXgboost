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
import format_process
import train_model
from count_name import replace_str
from get_train_data import compare_paper_with_set
import multiprocessing
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

def load_nltk_result():
    with open('data/track2/test/test_pub_nltk_result_title.res', 'rb') as r1:
        nltk_title = pickle.load(r1)
    with open('data/track2/test/test_pub_nltk_result_abstract.res', 'rb') as r3:
        nltk_abstract = pickle.load(r3)
    return nltk_title, nltk_abstract


def load_gensim_result():
    with open('data/track2/test/test_pub_gensim_title.json', 'rb') as r1:
        gensim_title = pickle.load(r1)
    with open('data/track2/test/test_pub_gensim_abstract.json', 'rb') as r3:
        gensum_abstract = pickle.load(r3)
    return gensim_title, gensum_abstract

def f(cna_valid_unass_competition, cna_valid_pub, test_alter_pub, model_name, model2_name, pool_id):
    print('pool_id', pool_id, 'begin')
    result_dict = {}
    error_times = 0
    bst = xgb.Booster(model_file=model_name)
    est = joblib.load(model2_name)
    if pool_id == 3:
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
                                        unass_paper_id, test_alter_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract)
                cna_x.append( x )
                id_list.append(same_name_author_id)

            dtest=xgb.DMatrix(np.array(cna_x))
            #ypred = bst.predict(dtest) + est.predict(np.array(cna_x))
            ypred =  est.predict(np.array(cna_x))
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
    model_name = 'xgb_1.model'
    model2_name = 'gdbt_1.model'
    with open('data/track2/cna_data/cna_valid_unass_competition.json', 'r') as r:
        cna_valid_unass_competition = json.load(r)
    with open('data/track2/cna_data/cna_valid_pub.json', 'r') as r:
        cna_valid_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile_pub.json', 'r') as r:
        whole_author_profile_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile.json', 'r') as r:
        whole_author_profile = json.load(r)

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
    if False:
        format_process.save_format_data(test_pub, file_name_alter_pub)
    #assert False

    #train_model
    if True:
        with open(file_name_alter_pub, 'r') as r:
            test_alter_pub = json.load(r)
        if False:
            data_model_dir = 'data/track2/test/test_pub_nltk_' + 'model_'
            data_result_dir = 'data/track2/test/test_pub_nltk_' + 'result_'
            train_model.nltk_idf(test_alter_pub, data_model_dir)
            train_model.nltk_tf(test_alter_pub, data_model_dir, data_result_dir, 
                                whole_data_hash_by_name, None, negative_example)

        if False:
            data_model_dir = 'data/track2/train/train_pub_gensim_' + 'model_'
            data_result_dir = 'data/track2/test/test_pub_gensim_' + 'result_'
            train_model.gensim_result(test_pub, data_model_dir, data_result_dir,
                                whole_data_hash_by_name, None, negative_example)
    
    nltk_title, nltk_abstract = load_nltk_result()
    #gensim_title, gensum_abstract = load_gensim_result()
    gensim_title, gensum_abstract = None, None

    #cna_valid_unass_competition = cna_valid_unass_competition[:20]
    num_pool = 4
    len_data = len(cna_valid_unass_competition)
    print("Length of data", len_data)
    pool = Pool()#train_pub
    step = len_data//num_pool
    id = 0
    sub_data = []
    jobs = []
    for one_data in cna_valid_unass_competition:
        sub_data.append(one_data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f, args=(sub_data, cna_valid_pub, test_alter_pub, model_name,model2_name, id)))
            id += 1
            sub_data = []

    if len(sub_data) > 0:
        jobs.append(pool.apply_async(f, args=(sub_data, cna_valid_pub, test_alter_pub, model_name,model2_name, id)))
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

