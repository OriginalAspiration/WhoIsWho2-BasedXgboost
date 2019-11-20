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


global corpus_title, corpus_abstract, model_title, model_abstract

# 返回一个五维向量:
# [ x1, x2, x3, x4, x5, ...]
#   x1: 是否有相同的协作者 -> 有两个相同的协作者
#   x2: 是否来自相同的单位
#   x3: 是否来自相同的出版社 -> 直接对比，是不是可以处理一下
#   x4: 年份差距
#   x5: 关键词->是否有相同的关键词
#   x6: title
#   x7: abstract摘要相似度


def add_variate_same_author(result, docs, paper_id_1, paper_id_2):
    try:
        same_author = 0
        for author_1 in docs[paper_id_1]['authors']:
            for author_2 in docs[paper_id_2]['authors']:
                if author_1['name'] == author_2['name']:
                    same_author += 1
                    break
            if same_author >= 2:
                break
        if same_author >= 2:
            result.append(1)  # has same collaborator
        else:
            result.append(0)
    except:
        result.append(0)


# 判断是否是相同组织
def add_variate_same_org(result, docs, paper_id_1, paper_id_2, author_rank):
    try:
        the_author_name = docs[paper_id_1]['authors'][author_rank]['name']
        same_org = 0
        for author_2 in docs[paper_id_2]['authors']:
            if replace_str(author_2['name']) == the_author_name:
                if author_2['org'] == docs[paper_id_1]['authors'][author_rank]['org']:
                    same_org += 1
                    break
        if same_org >= 1:
            result.append(1)  # has same org
        else:
            result.append(0)
    except:
        result.append(0)


def add_nltk_title(result, paper_id_1, paper_id_2):
    global nltk_title
    try:
        tup = (paper_id_1, paper_id_2)
        result.append(nltk_title[tup])
    except:
        result.append(0)


def add_nltk_abstract(result, paper_id_1, paper_id_2):
    global nltk_abstract
    try:
        tup = (paper_id_1, paper_id_2)
        result.append(nltk_abstract[tup])
    except:
        result.append(0)


def add_gensim_title(result, paper_id_1, paper_id_2):
    global gensim_title
    try:
        tup = (paper_id_1, paper_id_2)
        result.append(gensim_title[tup])
    except:
        result.append(0)


def add_gensim_abstract(result, paper_id_1, paper_id_2):
    global gensim_abstract
    try:
        tup = (paper_id_1, paper_id_2)
        result.append(gensim_abstract[tup])
    except:
        result.append(0)


def compare_two_paper(docs, paper_id_1, paper_id_2, author_rank):
    result = []
    # the_author_name = paper_info_1['authors'][author_rank]['name']
    # 第一维度

    add_variate_same_author(result,
                            docs,
                            paper_id_1,
                            paper_id_2)
    # 第二维度：是否来自相同的单位
    # todo : 这里是直接比较相等，其实名称还需要处理一下
    add_variate_same_org(result,
                         docs,
                         paper_id_1,
                         paper_id_2,
                         author_rank)
    # 第三维度：是否来自相同的出版
    try:
        if docs[paper_id_1]['venue'] == docs[paper_id_2]['venue']:
            result.append(1)  # has same venue
        else:
            result.append(0)
    except:
        result.append(0)
    # 第四维度：年份
    try:
        result.append(abs(docs[paper_id_1]['year'] - docs[paper_id_2]['year']))
        # year gap between two papers
    except:
        result.append(0)
    # 第五维度：关键词
    try:
        same_keyword = 0
        for keyword_1 in docs[paper_id_1]['keywords']:
            for keyword_2 in docs[paper_id_2]['keywords']:
                if keyword_1 == keyword_2:
                    # if fp.compare_two_keywords(keyword_1, keyword_2):
                    same_keyword += 1
                    break
            if same_keyword >= 1:
                break
        if same_keyword >= 1:
            result.append(1)  # has same keyword
        else:
            result.append(0)
    except:
        result.append(0)
    # 第六维度 title相似度
    add_nltk_title(result, paper_id_1, paper_id_2)
    # 第七维度 abstract相似度
    add_nltk_abstract(result, paper_id_1, paper_id_2)
    # 第8维：gensim doc2vec
    add_gensim_title(result, paper_id_1, paper_id_2)
    # 第9维：gensim doc2vec
    add_gensim_abstract(result, paper_id_1, paper_id_2)
    return result


def replace_str(input):
    return input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()


def load_nltk_result():
    global nltk_title, nltk_abstract
    with open('data/track2/train/train_pub_nltk_result_title.json', 'rb') as r1:
        nltk_title = pickle.load(r1)
    with open('data/track2/train/train_pub_nltk_result_abstractr.json', 'rb') as r3:
        nltk_abstract = pickle.load(r3)


def load_gensim_result():
    global gensim_title, gensum_abstract
    data_dir = "data/track2/train/train_pub_gensim_"
    with open('data/track2/train/train_pub_gensim_result_title.json', 'rb') as r1:
        gensim_title = pickle.load(r1)
    with open('data/track2/train/train_pub_gensim_result_abstractr.json', 'rb') as r3:
        gensum_abstract = pickle.load(r3)


if __name__ == "__main__":
    with open('data/track2/train/train_unass_data.json', 'r') as r:
        train_unass_data = json.load(r)
    with open('data/track2/train/train_existing_data.json', 'r') as r:
        train_existing_data = json.load(r)
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)
    load_nltk_result()
    load_gensim_result()
    existing_data_hash_by_name = {}
    for person_id in train_existing_data:
        real_name = train_existing_data[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in existing_data_hash_by_name:
            existing_data_hash_by_name[replaced_real_name] = {}
        existing_data_hash_by_name[replaced_real_name][person_id] = train_existing_data[person_id]['papers']

    len_train_unass_data = len(train_unass_data)
    print("The number of Train_unass_data:", len_train_unass_data)
    train_x = []
    train_y = []
    total = 0

    # 随机种子
    random.seed(2333)
    for unass_data in tqdm(train_unass_data):
        # print("[Unass_data ", total, "/", len_train_unass_data, "]")
        # total += 1
        unass_paper_id = unass_data[0][:8]
        author_rank = int(unass_data[0][9:])
        unass_author_id = unass_data[1]
        unass_paper_info = train_pub[unass_paper_id]
        the_author_name = replace_str(unass_paper_info['authors'][author_rank]['name'])
        # 正样本：同名作者
        for same_name_author_id in existing_data_hash_by_name[the_author_name]:
            if same_name_author_id == unass_author_id:
                one_person_sim_list = []
                for paper_id in existing_data_hash_by_name[the_author_name][same_name_author_id]:
                    one_person_sim_list.append(
                        compare_two_paper(train_pub,
                                          unass_paper_id,
                                          paper_id,
                                          author_rank))
                # print(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
                train_x.append(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
                train_y.append(1)
        # 随机负样本： 不同id作者
        if len(existing_data_hash_by_name[the_author_name]) > 1:
            diff_name_author_id = list(existing_data_hash_by_name[the_author_name])
            diff_name_author_id.remove(unass_author_id)
            # print(diff_name_author_id)
            same_name_author_id = random.choice(diff_name_author_id)
            one_person_sim_list = []
            for paper_id in existing_data_hash_by_name[the_author_name][same_name_author_id]:
                one_person_sim_list.append(
                    compare_two_paper(train_pub,
                                      unass_paper_id,
                                      paper_id,
                                      author_rank))
            # print(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
            train_x.append(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
            train_y.append(0)

with open('data/track2/train/train_x.pkl', 'wb') as wb:
    pickle.dump(np.array(train_x), wb)
with open('data/track2/train/train_y.pkl', 'wb') as wb:
    pickle.dump(np.array(train_y), wb)
