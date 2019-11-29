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
import re
import nltk.stem.porter as pt
from nltk.corpus import stopwords
from train_model import cosVector

def fix_name(s):
    s = s.lower().strip()
    x = re.split(r'[^a-z]', s)
    set_x = set()
    for a in x:
        if len(a) > 0:
            set_x.add(a)
    x = list(set_x)
    x.sort()
    s = ''.join(x)
    return s

#cnt1 = 0
#cnt2 = 0

def compress_name(name):
    #print('name', name)
    name = name.lower().strip()
    x = re.split(r'[^a-z]', name)
    word_list = [word for word in x if len(word) > 0]
    if len(word_list) == 0:
        return '', False
    is_compress = False
    if len(word_list[0]) == 1:
        is_compress = True
    
    compress_word_list = []
    LEN = len(word_list)
    for i in range(LEN):
        if i == LEN-1:
            compress_word_list.append(word_list[i])
        else:
            compress_word_list.append(word_list[i][0])
    return '_'.join(compress_word_list), is_compress

def add_variate_same_author2(result, paper_info_1, paper_info_2):
    #global cnt1 , cnt2
    
    same_author3 = 0
    is_compress_authors2 = set()
    no_compress_authors2 = set()
    authors2 = set()
    for author_2 in paper_info_2['authors']:
        author_2_name = replace_str(author_2['name'])
        if len(author_2_name) > 0:
            authors2.add( author_2_name )
        
        author_2_name = fix_name(author_2['name'])
        if len(author_2_name) > 0:
            authors2.add( author_2_name )
        
        author_2_name, is_compress = compress_name(author_2['name'])
        if len(author_2_name) > 0:
            if is_compress:
                is_compress_authors2.add(author_2_name)
            else:
                no_compress_authors2.add(author_2_name)

    for author_1 in paper_info_1['authors']:
        author_1_name, is_compress = compress_name(author_1['name'])
        flag = False
        
        if replace_str(author_1['name']) in authors2 or \
            fix_name(author_1['name']) in authors2:
            flag = True

        if flag == False:
            if is_compress:
                if author_1_name in no_compress_authors2:
                    flag = True
            else:
                if author_1_name in is_compress_authors2:
                    flag = True
        
        if flag:
            same_author3 += 1
    
    if same_author3 >= 2:
        result.append(1)  # has same collaborator
    else:
        result.append(0)
    # the same author may be > 1000
    result.append( min(same_author3,20)*1.0/20 )
    

    if same_author3 == 0:
        result.append(0)
    else:
        precision = same_author3*1.0/len(paper_info_1['authors'])
        recall = same_author3*1.0/len(paper_info_2['authors'])
        F1 = 2*precision*recall/(precision+recall)
        result.append(F1)

def compare_name(name1, name2):
    if replace_str(name1) == replace_str(name2):
        return True
    if fix_name(name1) == fix_name(name2):
        return True
    n1, i1 = compress_name(name1)
    n2, i2 = compress_name(name2)

    if (i1 is False or i2 is False) and n1 == n2:
        return True
    return False

def F1_between_two_org(org1, org2):
    list1 = re.split(r'[^a-zA-Z]', org1.lower())
    word_set = set()
    len1 = 0
    xx = []
    for word in list1:
        pt_stem = pt_stemmer.stem(word)
        if len(pt_stem) > 0 and  pt_stem not in stop_words:
            xx.append(pt_stem)
            word_set.add(pt_stem)
            len1 += 1

    list2 = re.split(r'[^a-zA-Z]', org2.lower())
    len2 = 0
    same_word = 0
    yy = []
    for word in list2:
        pt_stem = pt_stemmer.stem(word)
        if len(pt_stem) > 0 and  pt_stem not in stop_words:
            yy.append(pt_stem)
            len2 += 1
            if pt_stem in word_set:
                same_word += 1
    
    if same_word:
        precision = same_word*1.0/len1
        recall = same_word*1.0/len2
        F1 = 2*precision*recall/(precision+recall)
    else:
        F1 = 0
    
    return F1
 
def add_variate_same_org2(result, paper_info_1, paper_info_2, author_rank):
    same_org = 0
    orgs = []
    for author_2 in paper_info_2['authors']:
        if compare_name(author_2['name'], paper_info_1['authors'][author_rank]['name']):
            if 'org' in author_2 and 'org' in paper_info_1['authors'][author_rank]:
                if replace_str(author_2['org']) == replace_str(paper_info_1['authors'][author_rank]['org']):
                    same_org = 1
                else:
                    same_org = max(same_org, F1_between_two_org(author_2['org'], paper_info_1['authors'][author_rank]['org']) )
                    orgs.append(author_2['org'])
                    '''print(author_2['org'])
                    print(paper_info_1['authors'][author_rank]['org'])
                    print(replace_str(author_2['org']))
                    print(replace_str(paper_info_1['authors'][author_rank]['org']))
                    print('=================================================')'''

    result.append(same_org)

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

pt_stemmer = pt.PorterStemmer()
stop_words = set(stopwords.words('english'))
# add punctuations
punctuations = list(string.punctuation)
[stop_words.add(punc) for punc in punctuations]
# remove null
stop_words.add("null")
stop_words.add("journal")
stop_words.add("univers")
stop_words.add("laboratori")
stop_words.add("key")
stop_words.add("journal")
stop_words.add("school")
stop_words.add("institut")
stop_words.add("state")
stop_words.add("china")
stop_words.add("chines")
stop_words.add("lab")
stop_words.add("center")
stop_words.add("scienc")
stop_words.add("colleg")
stop_words.add("depart")
stop_words.add("intern")
stop_words.add("advanc")

def add_variate_same_venue2(result, paper_info_1, paper_info_2):
    #global cnt1 , cnt2
    #cnt2 += 1
    if replace_str(paper_info_1['venue']) == replace_str(paper_info_2['venue']):
        result.append(1)  # has same venue
    else:
        #result.append(0)

        list1 = re.split(r'[^a-zA-Z]', paper_info_1['venue'].lower())
        word_set = set()
        len1 = 0
        xx = []
        for word in list1:
            pt_stem = pt_stemmer.stem(word)
            if len(pt_stem) > 0 and  pt_stem not in stop_words:
                xx.append(pt_stem)
                word_set.add(pt_stem)
                len1 += 1

        list2 = re.split(r'[^a-zA-Z]', paper_info_2['venue'].lower())
        len2 = 0
        same_word = 0
        yy = []
        for word in list2:
            pt_stem = pt_stemmer.stem(word)
            if len(pt_stem) > 0 and  pt_stem not in stop_words:
                yy.append(pt_stem)
                len2 += 1
                if pt_stem in word_set:
                    same_word += 1
        
        if same_word:
            precision = same_word*1.0/len1
            recall = same_word*1.0/len2
            F1 = 2*precision*recall/(precision+recall)
        else:
            F1 = 0
            
        '''if F1 > 0:
            print(paper_info_1['venue'])
            print(paper_info_2['venue'])
            print('F1', F1)
            print('xx', xx)
            print('yy', yy)

            #cnt1 += 1
            #print('cnt1', cnt1, 'cnt2', cnt2)
        else:
            print('==='=======================')
            print('')'''
        
        result.append(F1)

def add_variate_same_keywords2(result, paper_info_1, paper_info_2):
    same_keyword2 = 0
    keyword_list1 = []
    for keyword_1 in paper_info_1['keywords']:
        w = ''.join([pt_stemmer.stem( re.sub(r'[^a-zA-Z]', '', word.lower() ) ) for word in keyword_1])
        if len(w) > 0:
            keyword_list1.append(w)

    keyword_list2 = []
    for keyword_2 in paper_info_2['keywords']:
        w = ''.join([pt_stemmer.stem( re.sub(r'[^a-zA-Z]', '', word.lower() ) ) for word in keyword_2])
        if len(w) > 0:
            keyword_list2.append(w)
    
    for w1 in keyword_list1:
        for w2 in keyword_list2:
            if w1 == w2:
                same_keyword2 += 1

    if same_keyword2 >= 1:
        result.append(1)  # has same keyword
    else:
        result.append(0)
    result.append( min(5.0, same_keyword2) / 5 )

    if same_keyword2 == 0:
        result.append(0)
    else:
        precision = same_keyword2*1.0/len(keyword_list1)
        recall = same_keyword2*1.0/len(keyword_list2)
        F1 = 2*precision*recall/(precision+recall)
        result.append(F1)

def add_variate_year2(result, paper_info_1, paper_info_2):
    if 'year' not in paper_info_1 or "year" not in paper_info_2 or paper_info_1['year'] == "" or paper_info_2['year'] == "" or paper_info_1['year'] == 0 or paper_info_2['year'] == 0:
        result.append( 5.0/20 )
        # t = 0, t = 1 , t > 5
        result.append( 0 )
        result.append( 0 )
        result.append( 0.5 )
    else:
        try:
            t = abs(paper_info_1['year'] - paper_info_2['year'])
            result.append( min(t, 20)*1.0/20 )

            result.append( t == 0 )
            result.append( t == 1 )
            result.append( t > 5 )
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

    t = np.sum((vector_1 - vector_2) ** 2)
    
    t = min(t, 10.0) / 10.0

    result.append(t)

    if math.isnan(float(t)):
        print('vector_1', vector_1)
        print('vector_2', vector_2)

        assert False

def add_two_hop_keyword(result, paper_info_1, paper_info_2, k2k_edges):
    kw_list_1 = [process_keyword(word) for word in paper_info_1['keywords']]
    kw_list_2 = [process_keyword(word) for word in paper_info_2['keywords']]
    n = 0
    for kw1 in kw_list_1:
        if kw1 not in k2k_edges:
            continue
        for kw2 in kw_list_2:
            if kw2 in k2k_edges[kw1]:
                n += 1
    result.append(n / (len(kw_list_1) * len(kw_list_2) + 1e-8))


def compare_two_paper(paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank,
                      nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result,
                      kdd_data=None, kdd_data_triplet=None, k2k_edges=None):
    result = []
    add_variate_same_author2(result, paper_info_1, paper_info_2)  # 3

    add_variate_same_org2(result, paper_info_1, paper_info_2, author_rank)  # 3
    
    add_variate_year2(result, paper_info_1, paper_info_2)  # 4

    add_variate_same_venue2(result, paper_info_1, paper_info_2) # 1

    add_variate_same_keywords2(result, paper_info_1, paper_info_2) # 3

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
    if k2k_edges is not None:
        add_two_hop_keyword(result, paper_info_1, paper_info_2, k2k_edges) # 1
    # 23
    return result

def compare_paper_with_set(id_list, unass_paper_id, train_pub, author_rank, nltk_title, nltk_abstract,
                           gensim_title, gensum_abstract, p2p_result, kdd_data=None,
                           kdd_data_triplet=None, k2k_edges=None):
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
                                        kdd_data, kdd_data_triplet, k2k_edges))
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

def f(negative_example, existing_data_hash_by_name, train_pub, kdd_data, kdd_data_triplet, k2k_edges, pool_id, is_negative):
    print('pool_id', pool_id, 'begin')
    results = []

    if pool_id == 0:
        x_negative_example = tqdm(negative_example)
    else:
        x_negative_example = negative_example
    
    if is_negative:
        for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in x_negative_example:
            x = compare_paper_with_set(existing_data_hash_by_name[the_author_name][other_name_author_id], unass_paper_id, 
                                    train_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result,
                                    kdd_data, kdd_data_triplet, k2k_edges)
            results.append(x)
    else:
        for unass_author_id, unass_paper_id, the_author_name, author_rank in x_negative_example:
            x = compare_paper_with_set(existing_data_hash_by_name[the_author_name][unass_author_id], unass_paper_id, 
                                    train_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract, p2p_result,
                                    kdd_data, kdd_data_triplet, k2k_edges)
            results.append(x)

    return results

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)
    nltk_title, nltk_abstract = load_nltk_result()
    #gensim_title, gensum_abstract = None, None
    gensim_title, gensum_abstract = load_gensim_result()
    p2p_result = load_p2p_result()
    # p2p_result = None
    #for x in p2p_result:
    #    if x[0] == 'AriXov6L':
    #        print(x, p2p_result[x])
    #assert False

    train_x = []
    train_y = []
    total = 0


    with open('data/track2/train/training_data.pkl', 'rb') as file:
        existing_data_hash_by_name,positive_example,negative_example = pickle.load(file)

    #with open('data/kdd_embedding/train_pid_order_to_features.pkl', 'rb') as rb:
    #    kdd_data = pickle.load(rb)
    #with open('data/kdd_embedding/train_pid_order_to_features_triplet.pkl', 'rb') as rb:
    #    kdd_data_triplet = pickle.load(rb)
    kdd_data = None
    kdd_data_triplet = None
    with open('data/track2/train/keywords_map.pkl', 'rb') as rb:
        k2k_edges = pickle.load(rb)

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
    mean_x = []
    '''sum_t_100 = 0
    sum_t_64 = 0
    count = 0'''

    num_pool = 4
    len_data = len(positive_example)
    print("Length of data", len_data)
    pool = Pool()#train_pub
    step = len_data//num_pool
    id = 0
    sub_data = []
    jobs = []
    for one_data in positive_example:
        sub_data.append(one_data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f, args=(sub_data, existing_data_hash_by_name, 
                                train_pub, kdd_data, kdd_data_triplet, k2k_edges, id, False)))
            id += 1
            sub_data = []

    if len(sub_data) > 0:
        jobs.append(pool.apply_async(f, args=(sub_data, existing_data_hash_by_name, 
                                    train_pub, kdd_data, kdd_data_triplet, k2k_edges, id, False)))
        id += 1
        sub_data = {}
    
    pool.close()
    pool.join()

    for j in jobs:
        sub_results = j.get()
        for x in sub_results:
            train_x.append(x)
            train_y.append(1)

            if min_x is None:
                min_x = x
                max_x = x
            min_x = np.min( [x, min_x], axis=0 )
            max_x = np.max( [x, max_x], axis=0 )
            mean_x.append(x)
    
    print('min_x', min_x)
    print('max_x', max_x)
    print('mean_x', np.mean(mean_x, axis=0))
    
    #assert False 6031508
    
    '''sum_t_100 = 0
    sum_t_64 = 0
    count = 0'''
    
    min_x = None
    max_x = None
    mean_x = []

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
            jobs.append(pool.apply_async(f, args=(sub_data, existing_data_hash_by_name, 
                                        train_pub, kdd_data, kdd_data_triplet, k2k_edges, id, True)))
            id += 1
            sub_data = []

    if len(sub_data) > 0:
        jobs.append(pool.apply_async(f, args=(sub_data, existing_data_hash_by_name, 
                                        train_pub, kdd_data, kdd_data_triplet, k2k_edges, id, True)))
        id += 1
        sub_data = {}
    
    pool.close()
    pool.join()

    for j in jobs:
        sub_results = j.get()
        for x in sub_results:
            train_x.append(x)
            train_y.append(0)
            if min_x is None:
                min_x = x
                max_x = x
            min_x = np.min( [x, min_x], axis=0 )
            max_x = np.max( [x, max_x], axis=0 )
            mean_x.append(x)
    
    print('min_x', min_x)
    print('max_x', max_x)
    print('mean_x', np.mean(mean_x, axis=0))
    
    '''print('sum_t_100, sum_t_64, count, sum_t_100/count, sum_t_64/count',
        sum_t_100, sum_t_64, count, sum_t_100/count, sum_t_64/count)'''

    with open('data/track2/train/train_x.pkl', 'wb') as wb:
        pickle.dump(np.array(train_x), wb)
    with open('data/track2/train/train_y.pkl', 'wb') as wb:
        pickle.dump(np.array(train_y), wb)
