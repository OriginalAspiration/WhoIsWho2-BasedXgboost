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

import numpy as np

import format_process as fp


global word_dict, word_dict_abstract, corpus_title, corpus_abstract

# 返回一个五维向量:
# [ x1, x2, x3, x4, x5, ...]
#   x1: 是否有相同的协作者 -> 有两个相同的协作者
#   x2: 是否来自相同的单位
#   x3: 是否来自相同的出版社 -> 直接对比，是不是可以处理一下
#   x4: 年份差距
#   x5: 关键词->是否有相同的关键词
#   x6: title
#   x7: abstract摘要相似度


def cosVector(x, y):
    if len(x) != len(y):
        print('error input,x and y is not in the same space')
        return
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i]*y[i]   #sum(X*Y)
        result2 += x[i]**2     #sum(X*X)
        result3 += y[i]**2     #sum(Y*Y)
    #print(result1)
    #print(result2)
    #print(result3)
    # print("result is "+str(result1/((result2*result3)**0.5))) #结果显示
    ans = result1/((result2*result3)**0.5)
    print("result is " + str(ans))  # 结果显示
    return ans


def add_variate_same_author(result, paper_info_1, paper_info_2):
    try:
        same_author = 0
        for author_1 in paper_info_1['authors']:
            for author_2 in paper_info_2['authors']:
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


def add_variate_same_org(result, paper_info_1, paper_info_2):
    try:
        same_org = 0
        for author_2 in paper_info_2['authors']:
            if author_2['name'] == the_author_name:
                if author_2['org'] == paper_info_1['authors'][author_rank]['org']:
                    same_org += 1
                    break
        if same_org >= 1:
            result.append(1)  # has same org
        else:
            result.append(0)
    except:
        result.append(0)


def get_doc_vector(doc):
    dict_size = len(word_dict)
    doc_vector = [0 for i in range(dict_size)]
    for word in doc.split():
        if word not in word_dict:
            continue
        word_index = word_dict[word]
        doc_vector[word_index] = corpus_title.tf_idf(word, doc)
    return doc_vector


def pre_doc_vector_(doc):
    global word_dict
    word_dict = {}
    total_word = 0
    for word in doc.split():
        if word not in word_dict:
            word_dict[word] = total_word
            total_word += 1


def get_doc_vector_abstract(doc):
    dict_size = len(word_dict)
    doc_vector = [0 for i in range(dict_size)]
    for word in doc.split():
        if word not in word_dict_abstract:
            continue
        word_index = word_dict_abstract[word]
        doc_vector[word_index] = corpus_abstract.tf_idf(word, doc)
    return doc_vector


def add_variate_title(result, paper_info_1, paper_info_2):
    try:
        title_vector_1 = get_doc_vector(paper_info_1['title'])
        title_vector_2 = get_doc_vector(paper_info_2['title'])
        ans = cosVector(title_vector_1, title_vector_2)
        result.append(ans)
        # print("Cos:",ans)
    except:
        result.append(0)
    return
#
#
def add_variate_title_abstract(result, paper_info_1, paper_info_2):
    try:
        title_vector_1 = get_doc_vector_abstract(paper_info_1['abstract'] + paper_info_1['title'])
        title_vector_2 = get_doc_vector_abstract(paper_info_2['abstract'] + paper_info_2['title'])
        ans = cosVector(title_vector_1, title_vector_2)
        result.append(ans)
        # print("Cos:",ans)
    except:
        result.append(0)
    return
#
#
#


def compare_two_paper(paper_info_1, paper_info_2, author_rank):
    result = []
    try:
        the_author_name = paper_info_1['authors'][author_rank]['name']
        # 第一维度
        add_variate_same_author(result, paper_info_1, paper_info_2)
        # 第二维度：是否来自相同的单位
        # todo : 这里是直接比较相等，其实名称还需要处理一下
        add_variate_same_org(result, paper_info_1, paper_info_2)
        # 第三维度：是否来自相同的出版
        try:
            if paper_info_1['venue'] == paper_info_2['venue']:
                result.append(1)  # has same venue
            else:
                result.append(0)
        except:
            result.append(0)
        # 第四维度：年份
        try:
            result.append(abs(paper_info_1['year'] - paper_info_2['year']))  # year gap between two papers
        except:
            result.append(0)
        # 第五维度：关键词
        try:
            same_keyword = 0
            for keyword_1 in paper_info_1['keywords']:
                for keyword_2 in paper_info_2['keywords']:
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
        add_variate_title(result, paper_info_1, paper_info_2)
        # 第七维度 title+abstract相似度
        add_variate_title_abstract(result, paper_info_1, paper_info_2)
    except:
        result = [0, 0, 0, 0, 0, 0]
    return result


def replace_str(input):
    return input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()


def load_nltk_result():
    global word_dict, word_dict_abstract, corpus_title, corpus_abstract
    with open('data/track2/train/train_tf_idf.txt', 'rb') as r1:
        corpus_title = pickle.load(r1)
    with open('data/track2/train/train_word_dict.txt', 'rb') as r2:
        word_dict = pickle.load(r2)
    with open('data/track2/train/train_abstract_tf_idf.txt', 'rb') as r3:
        corpus_abstract = pickle.load(r3)
    with open('data/track2/train/train_abstract_word_dict.txt', 'rb') as r4:
        word_dict_abstract = pickle.load(r4)
    print(word_dict)
    print(word_dict_abstract)


if __name__ == "__main__":
    with open('data/track2/train/train_unass_data.json', 'r') as r:
        train_unass_data = json.load(r)
    with open('data/track2/train/train_existing_data.json', 'r') as r:
        train_existing_data = json.load(r)
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)
    load_nltk_result()
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
    for unass_data in train_unass_data:
        print("[Unass_data ", total, "/", len_train_unass_data, "]")
        total += 1
        unass_paper_id = unass_data[0][:8]
        author_rank = int(unass_data[0][9:])
        unass_author_id = unass_data[1]
        unass_paper_info = train_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank]['name']
        for same_name_author_id in existing_data_hash_by_name[replace_str(the_author_name)]:
            one_person_sim_list = []
            for paper_id in existing_data_hash_by_name[replace_str(the_author_name)][same_name_author_id]:
                one_person_sim_list.append(compare_two_paper(unass_paper_info, train_pub[paper_id], author_rank))
            train_x.append(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
            if unass_author_id == same_name_author_id:
                train_y.append(1)
            else:
                train_y.append(0)

# with open('data/track2/train/train_x.pkl', 'wb') as wb:
#     pickle.dump(np.array(train_x), wb)
# with open('data/track2/train/train_y.pkl', 'wb') as wb:
#     pickle.dump(np.array(train_y), wb)
