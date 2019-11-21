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

import numpy as np
from tqdm import tqdm

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
    result.append(same_author)

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
    the_author_name = paper_info_1['authors'][author_rank]['name']
    same_org = 0
    for author_2 in paper_info_2['authors']:
        if replace_str(author_2['name']) == the_author_name:
            if replace_str(author_2['org']) == replace_str(paper_info_1['authors'][author_rank]['org']):
                same_org += 1
                
    if same_org >= 1:
        result.append(1)  # has same org
    else:
        result.append(0)
    result.append(same_org)



    same_org = 0
    
    orgs2 = set()
    for author_2 in paper_info_2['authors']:
        if 'org' in author_2 and len(replace_str(author_2['org'])) > 0:
            orgs2.add( replace_str(author_2['org']) )

    for author_1 in paper_info_1['authors']:
        if 'org' in author_1 and replace_str(author_1['org']) in orgs2:
            same_org += 1

    result.append(same_org)

    if same_org == 0:
        result.append(0)
    else:
        precision = same_org*1.0/len(paper_info_1['authors'])
        recall = same_org*1.0/len(paper_info_2['authors'])
        F1 = 2*precision*recall/(precision+recall)
        result.append(F1)

def add_lang_result(result, paper_id_1, paper_id_2, lang_result):
    if lang_result is None:
        f = 0
    else:
        tup = (paper_id_1, paper_id_2)
        f = lang_result[tup]
    result.append(f)
    result.append(np.log(f+1e-8))

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
    result.append(same_keyword)

    if same_keyword == 0:
        result.append(0)
    else:
        precision = same_keyword*1.0/len(paper_info_1['keywords'])
        recall = same_keyword*1.0/len(paper_info_2['keywords'])
        F1 = 2*precision*recall/(precision+recall)
        result.append(F1)

def add_variate_year(result, paper_info_1, paper_info_2):
    if 'year' not in paper_info_1 or "year" not in paper_info_2 or paper_info_1['year'] == "" or paper_info_2['year'] == "":
        result.append( 20 )
        #result.append( 100.0/50 )
        #result.append( np.log(100) )
    else:
        try:
            result.append(abs(paper_info_1['year'] - paper_info_2['year']))
            #result.append( abs(paper_info_1['year'] - paper_info_2['year'])*1.0/50 )
            #result.append( np.log(abs(paper_info_1['year'] - paper_info_2['year'] + 1e-8)) )
        except:
            print(paper_info_1['year'], paper_info_2['year'])
            print(type(paper_info_1['year']), type(paper_info_2['year']))
            assert False

def compare_two_paper(paper_id_1, paper_id_2, paper_info_1, paper_info_2, author_rank,
                                      nltk_title, nltk_abstract,
                                      gensim_title, gensum_abstract):
    result = []
    add_variate_same_author(result, paper_info_1, paper_info_2)

    add_variate_same_org(result, paper_info_1, paper_info_2, author_rank)
    
    add_variate_year(result, paper_info_1, paper_info_2)

    #add_variate_same_venue(result, paper_info_1, paper_info_2)

    #add_variate_same_keywords(result, paper_info_1, paper_info_2)
    
    #add_lang_result(result, paper_id_1, paper_id_2, nltk_title)
    #add_lang_result(result, paper_id_1, paper_id_2, nltk_abstract)
    return result

def compare_paper_with_set(id_list, unass_paper_id, train_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract):
    one_person_sim_list = []
    for paper_id in id_list:
        #print('paper_id',paper_id, 'unass_paper_id', unass_paper_id)
        one_person_sim_list.append(
                    compare_two_paper(unass_paper_id,
                                        paper_id,
                                        train_pub[unass_paper_id],
                                        train_pub[paper_id],
                                        author_rank,
                                        nltk_title, nltk_abstract,
                                        gensim_title, gensum_abstract))
    x = np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list)
    
    #x = np.concatenate([x, np.max(one_person_sim_list, axis=0)], 0)
    #x = np.concatenate([x, np.min(one_person_sim_list, axis=0)], 0)
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


if __name__ == "__main__":
    with open('data/track2/train/train_unass_data.json', 'r') as r:
        train_unass_data = json.load(r)
    with open('data/track2/train/train_existing_data.json', 'r') as r:
        train_existing_data = json.load(r)
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)
    nltk_title, nltk_abstract = load_nltk_result()
    gensim_title, gensum_abstract = None, None
    #gensim_title, gensum_abstract = load_gensim_result()

    len_train_unass_data = len(train_unass_data)
    print("The number of Train_unass_data:", len_train_unass_data)
    train_x = []
    train_y = []
    total = 0


    with open('data/track2/train/training_data.pkl', 'rb') as file:
        existing_data_hash_by_name,positive_example,negative_example = pickle.load(file)
    min_x = None
    max_x = None
    cnt = 0
    for unass_author_id, unass_paper_id, the_author_name, author_rank in tqdm(positive_example):
        x = compare_paper_with_set(existing_data_hash_by_name[the_author_name][unass_author_id], unass_paper_id, 
                                train_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract)
        train_x.append(x)
        train_y.append(1)

        if min_x is None:
            min_x = x
            max_x = x
        min_x = np.min([min_x, x], axis=0)
        max_x = np.max([max_x, x], axis=0)
        cnt += 1
        if cnt > 1000:
            break

    cnt = 0
    for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in tqdm(negative_example):
        x = compare_paper_with_set(existing_data_hash_by_name[the_author_name][other_name_author_id], unass_paper_id, 
                                train_pub, author_rank, nltk_title, nltk_abstract, gensim_title, gensum_abstract)
        train_x.append(x)
        train_y.append(0)

        min_x = np.min([min_x, x], axis=0)
        max_x = np.max([max_x, x], axis=0)
        cnt += 1
        if cnt > 1000:
            break
    
    print('min_x', min_x)
    print('max_x', max_x)

    with open('data/track2/train/train_x.pkl', 'wb') as wb:
        pickle.dump(np.array(train_x), wb)
    with open('data/track2/train/train_y.pkl', 'wb') as wb:
        pickle.dump(np.array(train_y), wb)

    #TODO  1.可能要修改特征维度 2.可能和test那边能配套上
