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
import math

from train_model import cosVector

def add_variate_same_author(result, paper_info_1, paper_info_2):
    same_author = 0
    
    authors2 = set()
    for author_2 in paper_info_2['authors']:
        if len(replace_str(author_2['name'])) > 0:
            authors2.add( replace_str(author_2['name']) )

    for author_1 in paper_info_1['authors']:
        if replace_str(author_1['name']) in authors2:
            same_author += 1
            
        
    if same_author >= 2:
        result.append(1)  # has same collaborator
    else:
        result.append(0)
    # the same author may be > 1000
    result.append( min(same_author,20)*1.0/20 )
    

    if same_author == 0:
        result.append(0)
    else:
        precision = same_author*1.0/len(paper_info_1['authors'])
        recall = same_author*1.0/len(paper_info_2['authors'])
        F1 = 2*precision*recall/(precision+recall)
        
        '''try:
            assert precision <= 1
            assert recall <= 1
            assert F1 <= 1
        except:
            print(paper_info_1['authors'])
            print(paper_info_2['authors'])
            print(precision, recall, F1)
            assert False
        '''
        
        result.append(F1)


# 判断是否是相同组织 
def add_variate_same_org(result, paper_info_1, paper_info_2, author_rank):
    the_author_name = replace_str(paper_info_1['authors'][author_rank]['name'])
    same_org = 0
    for author_2 in paper_info_2['authors']:
        if replace_str(author_2['name']) == the_author_name:
            if 'org' in author_2 and 'org' in paper_info_1['authors'][author_rank] and \
                replace_str(author_2['org']) == replace_str(paper_info_1['authors'][author_rank]['org']):
                same_org += 1
                
    if same_org >= 1:
        result.append(1)  # has same org
    else:
        result.append(0)



    same_org = 0
    
    orgs2 = set()
    for author_2 in paper_info_2['authors']:
        if 'org' in author_2 and len(replace_str(author_2['org'])) > 0:
            orgs2.add( replace_str(author_2['org']) )

    for author_1 in paper_info_1['authors']:
        if 'org' in author_1 and replace_str(author_1['org']) in orgs2:
            same_org += 1

    result.append( min(same_org,20) * 1.0/20 )

    if same_org == 0:
        result.append(0)
    else:
        precision = same_org*1.0/len(paper_info_1['authors'])
        #TODO there still some bug
        recall = min(1.0, same_org*1.0/len(paper_info_2['authors']))
        F1 = 2*precision*recall/(precision+recall)
        result.append(F1)

def add_lang_result(result, paper_id_1, paper_id_2, lang_result, out_log=True):
    if lang_result is None:
        f = 0
    else:
        tup = (paper_id_1, paper_id_2)
        f = lang_result[tup]

    result.append(f)
    if out_log:
        t = max(0.0, -np.log(f+1e-8) )/20
        
        try:
            assert math.isnan(t) == False
            assert math.isnan(f) == False
        except:
            print('t', t, 'f', f)
            assert False
        result.append( t )


def add_variate_same_venue(result, paper_info_1, paper_info_2):
    if replace_str(paper_info_1['venue']) == replace_str(paper_info_2['venue']):
        result.append(1)  # has same venue
    else:
        result.append(0)

def add_variate_same_keywords(result, paper_info_1, paper_info_2):
    same_keyword = 0
    for keyword_1 in paper_info_1['keywords']:
        for keyword_2 in paper_info_2['keywords']:
            if replace_str( ''.join(keyword_1) ) == replace_str( ''.join(keyword_2) ):
                same_keyword += 1
                

    if same_keyword >= 1:
        result.append(1)  # has same keyword
    else:
        result.append(0)
    result.append( min(5.0, same_keyword) / 5 )

    if same_keyword == 0:
        result.append(0)
    else:
        precision = same_keyword*1.0/len(paper_info_1['keywords'])
        recall = same_keyword*1.0/len(paper_info_2['keywords'])
        F1 = 2*precision*recall/(precision+recall)
        result.append(F1)

def add_variate_year(result, paper_info_1, paper_info_2):
    if 'year' not in paper_info_1 or "year" not in paper_info_2 or paper_info_1['year'] == "" or paper_info_2['year'] == "":
        result.append( 1 )
    else:
        try:
            t = abs(paper_info_1['year'] - paper_info_2['year'])
            result.append( min(t, 20)*1.0/20 )
        except:
            print(paper_info_1['year'], paper_info_2['year'])
            print(type(paper_info_1['year']), type(paper_info_2['year']))
            assert False

def get_kdd_vector(paper_id, author_rank, kdd_data, vector_dims):
    key = paper_id + '-' + str(author_rank)
    if key in kdd_data.keys():
        vector = kdd_data[key]
    else:
        temp_vector_list = []
        for i in range(100):
            temp_key = paper_id + '-' + str(i)
            if temp_key not in kdd_data:
                break
            temp_vector_list.append(kdd_data[temp_key])
        if temp_vector_list:
            vector = np.sum(temp_vector_list, axis=0)/len(temp_vector_list)
        else:
            #return None
            print('Canot find it!')
            print('paper_id, author_rank', paper_id, author_rank)
            print('vector_dims', vector_dims)
            print('')
            
            vector = np.zeros(vector_dims)
    return vector + 1e-8

def add_variate_kdd(result, paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank_1, kdd_data, kdd_dims):
    #global sum_t_100, sum_t_64, count

    # get author_rank_2
    the_author_name = replace_str(paper_info_1['authors'][author_rank_1]['name'])
    author_rank_2 = -1
    for index, author_2 in enumerate(paper_info_2['authors']):
        if replace_str(author_2['name']) == the_author_name:
            author_rank_2 = index
            break
    vector_1 = get_kdd_vector(paper_id_1, author_rank_1, kdd_data, kdd_dims)
    vector_2 = get_kdd_vector(paper_id_2, author_rank_2, kdd_data, kdd_dims)
    
    '''if vector_1 is None or vector_2 is None:
        t = 0.0
    else:
        t = cosVector(vector_1, vector_2)'''

    t = np.sum((vector_1 - vector_2) ** 2)
    result.append(t)

    '''if kdd_dims == 100:
        sum_t_100 += t
    else:
        sum_t_64 += t
    count += 1'''

    if math.isnan(float(t)):
        print('vector_1', vector_1)
        print('vector_2', vector_2)

        assert False


def compare_two_paper(paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank,
                      nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result,
                      kdd_data=None, kdd_data_triplet=None):
    result = []
    add_variate_same_author(result, paper_info_1, paper_info_2)  # 3

    add_variate_same_org(result, paper_info_1, paper_info_2, author_rank)  # 3
    
    add_variate_year(result, paper_info_1, paper_info_2)  # 1

    add_variate_same_venue(result, paper_info_1, paper_info_2) # 1

    add_variate_same_keywords(result, paper_info_1, paper_info_2) # 3

    add_lang_result(result, paper_id_1, paper_id_2, nltk_title) # 2
    add_lang_result(result, paper_id_1, paper_id_2, nltk_abstract) # 2
    add_lang_result(result, paper_id_1, paper_id_2, gensim_title, False) # 1
    add_lang_result(result, paper_id_1, paper_id_2, gensum_abstract, False) # 1
    if p2p_result is not None:
        add_lang_result(result, paper_id_1, paper_id_2, p2p_result, False) # 1
    if kdd_data is not None:
        add_variate_kdd(result, paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank, kdd_data, 100) # 1
    if kdd_data_triplet is not None:
        add_variate_kdd(result, paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank, kdd_data_triplet, 64) # 1
    # 20
    return result

def compare_paper_with_set(id_list, unass_paper_id, train_pub, author_rank, nltk_title, nltk_abstract,
                           gensim_title, gensum_abstract, p2p_result, kdd_data=None, kdd_data_triplet=None):
    one_person_sim_list = []
    for paper_id in id_list:
        #print('paper_id',paper_id, 'unass_paper_id', unass_paper_id)
        if paper_id == unass_paper_id:
            continue
        one_person_sim_list.append(
                    compare_two_paper(unass_paper_id,
                                        paper_id,
                                        train_pub[unass_paper_id],
                                        train_pub[paper_id],
                                        author_rank,
                                        nltk_title, nltk_abstract,
                                        gensim_title, gensum_abstract, p2p_result,
                                        kdd_data, kdd_data_triplet))
    x = np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list)
    
    x = np.concatenate([x, np.max(one_person_sim_list, axis=0)], 0)
    x = np.concatenate([x, np.min(one_person_sim_list, axis=0)], 0)
    #print( np.shape(x) )
    #assert False
    return x

def replace_str(input):
    return input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()


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

def load_p2p_result():
    with open('data/track2/train/train_pub_p2p_result_title.res', 'rb') as r1:
        p2p_result = pickle.load(r1)
    return p2p_result

def f(negative_example, existing_data_hash_by_name, unass_paper_id, train_pub, 
        author_rank, kdd_data, kdd_data_triplet, pool_id):
    print('pool_id', pool_id, 'begin')
    results = []

    if pool_id == 0:
        x_negative_example = tqdm(negative_example)
    else:
        x_negative_example = negative_example
    for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in x_negative_example:
        x = compare_paper_with_set(existing_data_hash_by_name[the_author_name][other_name_author_id], unass_paper_id, 
                                train_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result,
                                kdd_data, kdd_data_triplet)
        results.append(x)

    return results

if __name__ == "__main__":
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)
    nltk_title, nltk_abstract = load_nltk_result()
    #gensim_title, gensum_abstract = None, None
    gensim_title, gensum_abstract = load_gensim_result()
    # p2p_result = load_p2p_result()
    p2p_result = None
    #for x in p2p_result:
    #    if x[0] == 'AriXov6L':
    #        print(x, p2p_result[x])
    #assert False

    train_x = []
    train_y = []
    total = 0


    with open('data/track2/train/training_data.pkl', 'rb') as file:
        existing_data_hash_by_name,positive_example,negative_example = pickle.load(file)

    with open('data/kdd_embedding/pid_order_to_features.pkl', 'rb') as rb:
        kdd_data = pickle.load(rb)
    with open('data/kdd_embedding/pid_order_to_features_triplet.pkl', 'rb') as rb:
        kdd_data_triplet = pickle.load(rb)

    '''max_t =0
    min_t =0
     
    for x in kdd_data_triplet:
        y = kdd_data_triplet[x]
        t = float(np.sum(y))
        
        if math.isnan(t) or t == 0:
            print('++', 'y', y, 'x', x)
        max_t = max(t, max_t)
        min_t = min(t, max_t)
    print('max_t', max_t)
    print('min_t', min_t)
    for x in kdd_data:
        y = kdd_data[x]
        t = float(np.sum(y))

        if math.isnan(t) or t == 0:
            print('y', y, 'x', x)
        
        max_t = max(t, max_t)
        min_t = min(t, max_t)
    print('max_t', max_t)
    print('min_t', min_t)
    
    assert False'''


    cnt = 0
    min_x = None
    max_x = None
    '''sum_t_100 = 0
    sum_t_64 = 0
    count = 0'''
    
    for unass_author_id, unass_paper_id, the_author_name, author_rank in tqdm(positive_example):
        x = compare_paper_with_set(existing_data_hash_by_name[the_author_name][unass_author_id], unass_paper_id, 
                                   train_pub, author_rank,  nltk_title, nltk_abstract, gensim_title, gensum_abstract,
                                   p2p_result, kdd_data, kdd_data_triplet)
        train_x.append(x)
        train_y.append(1)
        
        '''if count % 10000 == 0:
            print('sum_t_100, sum_t_64, count, sum_t_100/count, sum_t_64/count',
            sum_t_100, sum_t_64, count, sum_t_100/count, sum_t_64/count)'''

        
        if min_x is None:
            min_x = x
            max_x = x
        min_x = np.min( [x, min_x], axis=0 )
        max_x = np.max( [x, max_x], axis=0 )
    
    print('min_x', min_x)
    print('max_x', max_x)
    
    #assert False 6031508
    
    '''sum_t_100 = 0
    sum_t_64 = 0
    count = 0'''
    
    num_pool = 4
    len_data = len(negative_example)
    print("Length of data", len_data)
    pool = Pool()#train_pub
    step = len_data//num_pool
    id = 0
    sub_data = []
    jobs = []
    for one_data in negative_example:
        sub_data.append(one_data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f, args=(sub_data, existing_data_hash_by_name, unass_paper_id, 
                                train_pub, author_rank, kdd_data, kdd_data_triplet, id)))
            id += 1
            sub_data = []

    if len(sub_data) > 0:
        jobs.append(pool.apply_async(f, args=(sub_data, existing_data_hash_by_name, unass_paper_id, 
                    train_pub, author_rank, kdd_data, kdd_data_triplet, id)))
        id += 1
        sub_data = {}
    
    pool.close()
    pool.join()

    for j in jobs:
        sub_results = j.get()
        for x in sub_results:
            train_x.append(x)
            train_y.append(0)
    
    '''print('sum_t_100, sum_t_64, count, sum_t_100/count, sum_t_64/count',
        sum_t_100, sum_t_64, count, sum_t_100/count, sum_t_64/count)'''

    with open('data/track2/train/train_x.pkl', 'wb') as wb:
        pickle.dump(np.array(train_x), wb)
    with open('data/track2/train/train_y.pkl', 'wb') as wb:
        pickle.dump(np.array(train_y), wb)
