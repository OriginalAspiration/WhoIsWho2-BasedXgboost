
import string
import json
import pickle
import time
from tqdm import tqdm
import gensim
import random
from gensim import corpora, models, similarities
import os
import re
import math
import numpy as np
import multiprocessing
from multiprocessing import Pool

from scipy.spatial.distance import pdist


def cosVector(x, y):
    t = pdist(np.vstack([x, y]), 'cosine')[0]
    return float(1.0 - t)


def fix_name(s):
    s = s.lower().strip()
    x = re.split(r'[ \.\-\_]', s)
    set_x = set()
    for a in x:
        if len(a) > 0:
            set_x.add(a)
    x = list(set_x)
    x.sort()
    s = ''.join(x)
    return s


def get_doc_lsi_title(docs, dictionary, corpus_tfidf, corpus_lsi):
    vector_lsi_title = {}
    for data in tqdm(docs):
        query_bow = dictionary.doc2bow(docs[data]['title'].split())
        query_tfidf = corpus_tfidf[query_bow]
        query_lsi = corpus_lsi[query_tfidf]
        query_list = [val for id, val in query_lsi]
        vector_lsi_title[data] = query_list
    return vector_lsi_title


def get_doc_lsi_abstract(docs, dictionary, corpus_tfidf, corpus_lsi):
    vector_lsi_abstract = {}
    for data in tqdm(docs):
        query_bow = dictionary.doc2bow((docs[data]['title'] + ' ' + docs[data]['abstract']).split())
        query_tfidf = corpus_tfidf[query_bow]
        query_lsi = corpus_lsi[query_tfidf]
        query_list = [val for id, val in query_lsi]
        vector_lsi_abstract[data] = query_list
    return vector_lsi_abstract


def add_nltk_title(result, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    vector_1 = docs[paper_id_1]['lsiVecTitle']
    vector_2 = docs[paper_id_2]['lsiVecTitle']
    ans = cosVector(vector_1, vector_2)
    try:
        assert np.isnan(ans) == False
        assert math.isnan(ans) == False
        assert ans == ans
    except:
        print('vector_1', vector_1)
        print('vector_2', vector_2)
        print('ans', ans)
    result[tup] = ans
    return ans


def add_nltk_abstract(result, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    word_dict = {}
    # TODO it may be some bug for why "" in abstart. They's id is WSNj1aYr
    vector_1 = docs[paper_id_1]['lsiVecAbstract']
    vector_2 = docs[paper_id_2]['lsiVecAbstract']
    ans = cosVector(vector_1, vector_2)
    assert math.isnan(ans) == False
    result[tup] = ans
    return ans


def nltk_calc_two_paper(docs, paper_id_1, paper_id_2):
    '''if paper_id_1 not in docs or paper_id_2 not in docs:
        if paper_id_1 not in docs:
            print('xx paper_id_1', paper_id_1)
        if paper_id_2 not in docs:
            print('xx paper_id_2', paper_id_2)
        return 0, 0 , {(paper_id_1, paper_id_2): 0}, {(paper_id_1, paper_id_2): 0}
    else:'''
    result_title = {}
    result_abstract = {}
    ans1 = add_nltk_title(result_title,
                          docs,
                          paper_id_1, paper_id_2)
    ans2 = add_nltk_abstract(result_abstract,
                             docs,
                             paper_id_1, paper_id_2)
    return ans1, ans2, result_title, result_abstract


def nltk_train_lsi(docs):
    print("[train nltk lsi]")
    whole_title = []
    whole_abstract = []
    for data in tqdm(docs):
        if 'title' in docs[data]:
            whole_title.append(docs[data]['title'])
        if ('abstract' not in docs[data]) or (not docs[data]['abstract']):
            docs[data]['abstract'] = " "
        whole_abstract.append(docs[data]['abstract'])
    texts_title = [
        [word for word in document.split(" ")]
        for document in whole_title
    ]
    dictionary_title = corpora.Dictionary(texts_title)
    corpus_title = [dictionary_title.doc2bow(text) for text in texts_title]
    model_tfidf_title = models.TfidfModel(corpus_title)
    corpus_tfidf_title = model_tfidf_title[corpus_title]
    # corpus_tfidf_title.save(data_dir + 'tfidf_title.model')
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish tfidf title]")
    corpus_lsi_title = models.LsiModel(corpus_tfidf_title,
                                       id2word=dictionary_title,
                                       num_topics=50)  # 主题数，怎么设置？？
    # corpus_lsi_title.save(data_dir + 'lsi_title.model')
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lsi title ]")
    vector_lsi_title = get_doc_lsi_title(docs,
                                         dictionary_title,
                                         model_tfidf_title,
                                         corpus_lsi_title)

    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lsi vector ]")
    texts_abstract = [
        [word for word in document.split(" ")]
        for document in whole_abstract
    ]
    dictionary_abstract = corpora.Dictionary(texts_abstract)
    corpus_abstract = [dictionary_abstract.doc2bow(text) for text in texts_abstract]
    model_tfidf_abstract = models.TfidfModel(corpus_abstract)
    corpus_tfidf_abstract = model_tfidf_title[corpus_abstract]
    # corpus_tfidf_abstract.save(data_dir + 'tfidf_abstract.model')
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish tfidf abstract ]")
    corpus_lsi_abstract = models.LsiModel(corpus_tfidf_abstract,
                                          id2word=dictionary_abstract,
                                          num_topics=200)  # 主题数，怎么设置？？
    # corpus_lsi_abstract.save(data_dir + 'lsi_abstract.model')
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lsi abstract ]")
    vector_lsi_abstract = get_doc_lsi_abstract(docs,
                                               dictionary_title,
                                               model_tfidf_abstract,
                                               corpus_lsi_title)
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lsi vector]")
    return vector_lsi_title, vector_lsi_abstract



def f1(negative_example, existing_data_hash_by_name, pub,
       pool_id, is_negative=True):
    print('pool_id', pool_id, 'begin')
    list_ans1 = []
    list_ans2 = []
    result_title = {}
    result_abstract = {}
    if pool_id == 0:
        x = tqdm(negative_example)
    else:
        x = negative_example

    if is_negative:
        for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in x:
            for paper_id in existing_data_hash_by_name[the_author_name][other_name_author_id]:
                ans1, ans2, r1, r2 = nltk_calc_two_paper(pub,
                                                         unass_paper_id,
                                                         paper_id)
                result_title.update(r1)
                result_abstract.update(r2)
                list_ans1.append(ans1)
                list_ans2.append(ans2)
    else:
        for unass_author_id, unass_paper_id, the_author_name, author_rank in x:
            for paper_id in existing_data_hash_by_name[the_author_name][unass_author_id]:
                ans1, ans2, r1, r2 = nltk_calc_two_paper(pub,
                                                         unass_paper_id,
                                                         paper_id)
                result_title.update(r1)
                result_abstract.update(r2)
                list_ans1.append(ans1)
                list_ans2.append(ans2)

    print("pool", pool_id, "is finish.", "[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]",
          "[Finish lemma ]")
    return list_ans1, list_ans2, result_title, result_abstract


def nltk_get_sim(pub, data_model_dir, data_result_dir,
                 existing_data_hash_by_name,
                 positive_example,
                 negative_example):
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[begin nltk_lsi]")

    print("Get lsi vector")
    with open(data_model_dir + 'title.vec', 'rb') as fresult_title:
        vector_title = pickle.load(fresult_title)
    with open(data_model_dir + 'abstract.vec', 'rb') as fresult_abstract:
        vector_abstract = pickle.load(fresult_abstract)
    for data in tqdm(pub):
        pub[data]['lsiVecTitle'] = vector_title[data]
        pub[data]['lsiVecAbstract'] = vector_abstract[data]

    result_title = {}
    result_abstract = {}
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "begin calc compare")
    if positive_example is not None:
        num_pool = 1
        len_data = len(positive_example)
        print("Length of positive data", len_data)
        pool = Pool()
        step = len_data // num_pool

        id = 0
        sub_data = []
        jobs = []
        for data in positive_example:
            sub_data.append(data)
            if len(sub_data) >= step:
                jobs.append(pool.apply_async(f1, args=(
                    sub_data, existing_data_hash_by_name,
                    pub, id, False)))
                id += 1
                sub_data = []

        if len(sub_data) >= 0:
            jobs.append(pool.apply_async(f1, args=(
                sub_data, existing_data_hash_by_name,
                pub, id, False)))
            id += 1
            sub_data = []

        pool.close()
        pool.join()

        list_ans1 = []
        list_ans2 = []
        for j in jobs:
            a1, a2, r1, r2 = j.get()
            list_ans1 += a1
            list_ans2 += a2
            result_title.update(r1)
            result_abstract.update(r2)

        print('mean ans1', np.mean(list_ans1))
        print('mean ans2', np.mean(list_ans2))

    num_pool = 1
    len_data = len(negative_example)
    print("Length of negative data", len_data)
    pool = Pool()
    step = len_data // num_pool

    id = 0
    sub_data = []
    jobs = []
    for data in negative_example:
        sub_data.append(data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f1, args=(
                sub_data, existing_data_hash_by_name,
                pub, id)))
            id += 1
            sub_data = []

    if len(sub_data) >= 0:
        jobs.append(
            pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name,
                                       pub, id)))
        id += 1
        sub_data = []

    pool.close()
    pool.join()

    list_ans1 = []
    list_ans2 = []
    for j in jobs:
        a1, a2, r1, r2 = j.get()
        list_ans1 += a1
        list_ans2 += a2
        result_title.update(r1)
        result_abstract.update(r2)

    print('mean ans1', np.mean(list_ans1))
    print('mean ans2', np.mean(list_ans2))

    print(len(result_title))
    print(len(result_abstract))
    with open(data_result_dir + 'title.res', 'wb') as fresult_title:
        pickle.dump(result_title, fresult_title)

    with open(data_result_dir + 'abstract.res', 'wb') as fresult_abstract:
        pickle.dump(result_abstract, fresult_abstract)

    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[end nltk_lsi]")
    return None


def train_nltk_lsi_model(file_name_pub, file_name_out,
                         existing_data_hash_by_name,
                         positive_example,
                         negative_example):
    with open(file_name_pub, 'r') as r:
        pub = json.load(r)

    data_model_dir = file_name_out + 'vector_'
    # 训练模型
    if True:
    # if not os.path.exists(data_model_dir+'title.vec'):
        vector_lsi_title, vector_lsi_abstract = nltk_train_lsi(pub)
        with open(data_model_dir + 'title.vec', 'wb') as file:
            pickle.dump(vector_lsi_title, file)
        with open(data_model_dir + 'abstract.vec', 'wb') as file:
            pickle.dump(vector_lsi_abstract, file)
    # 预处理出结果,get_train_data阶段可以直接用
    data_result_dir = file_name_out + 'result_'
    nltk_get_sim(pub, data_model_dir, data_result_dir,
                 existing_data_hash_by_name,
                 positive_example,
                 negative_example)


if __name__ == "__main__":
    train_nltk_lsi = True
    #
    file_name_unass_data = 'data/track2/train/train_unass_data.json'
    file_name_existing_data = 'data/track2/train/train_existing_data.json'
    file_name_pub = 'data/track2/train/train_pub_alter.json'
    random.seed(2333)
    #
    positive_example = []
    negative_example = []
    existing_data_hash_by_name = {}

    if True:
    # if not os.path.exists('data/track2/train/training_data.pkl'):
        with open(file_name_unass_data, 'r') as r:
            unass_data = json.load(r)
        with open(file_name_existing_data, 'r') as r:
            existing_data = json.load(r)
        with open(file_name_pub, 'r') as r:
            pub = json.load(r)
        for person_id in existing_data:
            real_name = existing_data[person_id]['name']
            replaced_real_name = fix_name(real_name)
            if replaced_real_name not in existing_data_hash_by_name:
                existing_data_hash_by_name[replaced_real_name] = {}
            existing_data_hash_by_name[replaced_real_name][person_id] = existing_data[person_id]['papers']
        total = 0
        print('unass_data', len(unass_data))
        for unass_data in tqdm(unass_data):
            unass_paper_id = unass_data[0][:8]
            author_rank = int(unass_data[0][9:])
            unass_author_id = unass_data[1]
            unass_paper_info = pub[unass_paper_id]
            the_author_name = fix_name(unass_paper_info['authors'][author_rank]['name'])
            # 正样本：同名作者
            positive_example.append([unass_author_id, unass_paper_id, the_author_name, author_rank])
            # 随机负样本： 不同id作者
            if len(existing_data_hash_by_name[the_author_name]) > 1:
                diff_name_author_id = list(existing_data_hash_by_name[the_author_name])
                diff_name_author_id.remove(unass_author_id)
                other_name_author_id = random.choice(diff_name_author_id)
                negative_example.append(
                    [unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id])
        with open('data/track2/train/training_data.pkl', 'wb') as file:
            pickle.dump([existing_data_hash_by_name, positive_example, negative_example], file)

    out_file_name = 'data/track2/train/train_pub_nltk_lsi_'
    train_nltk_lsi_model(file_name_pub,
                         out_file_name,
                         existing_data_hash_by_name,
                         positive_example,
                         negative_example)
