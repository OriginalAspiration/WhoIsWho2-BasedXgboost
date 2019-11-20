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
import format_process as fp


def cosVector(x, y):
    if len(x) != len(y):
        print('error input,x and y is not in the same space')
        return 0
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i]*y[i]   #sum(X*Y)
        result2 += x[i]**2     #sum(X*X)
        result3 += y[i]**2     #sum(Y*Y)
    ans = result1/((result2*result3)**0.5)
    return ans


def pre_doc_vector(word_dict, doc):
    total_word = 0
    # print(doc)
    words = list(set(doc.split()))
    for word in words:
        if word not in word_dict:
            word_dict[word] = total_word
            total_word += 1


def get_doc_vector(word_dict, corpus, doc):
    dict_size = len(word_dict)
    doc_vector = [0 for i in range(dict_size)]
    words = list(set(doc.split()))
    for word in words:
        # print(word)
        word_index = word_dict[word]
        doc_vector[word_index] = corpus.tf_idf(word, doc)
    return doc_vector


def add_nltk_title(result, paper_info_1, paper_info_2):
    try:
        global nltk_title
        word_dict = {}
        pre_doc_vector(word_dict, paper_info_1['title'] + ' ' + paper_info_2['title'])
        vec_1 = get_doc_vector(word_dict, nltk_title, paper_info_1['title'])
        vec_2 = get_doc_vector(word_dict, nltk_title, paper_info_2['title'])
        ans = cosVector(vec_1, vec_2)
        result.append(ans)
    except:
        result.append(0)


def add_nltk_abstract(result, paper_info_1, paper_info_2):
    try:
        global nltk_abstract
        word_dict = {}
        pre_doc_vector(word_dict, paper_info_1['title'] + ' ' +
                                    paper_info_2['title'] + ' ' +
                                    paper_info_1['abstract'] + ' ' +
                                    paper_info_2['abstract'] + ' ')
        vec_1 = get_doc_vector(word_dict, nltk_abstract,
                paper_info_1['title'] + ' ' + paper_info_1['abstract'])
        vec_2 = get_doc_vector(word_dict, nltk_abstract,
                paper_info_2['title'] + ' ' + paper_info_2['abstract'])
        ans = cosVector(vec_1, vec_2)
        result.append(ans)
    except:
        result.append(0)


def add_gensim_title(result, paper_info_1, paper_info_2):
    global gensim_title
    try:
        vec_1 = gensim_title.infer_vector(paper_info_1['title'].split())
        vec_2 = gensim_title.infer_vector(paper_info_2['title'].split())
        ans = cosVector(vec_1, vec_2)
        result.append(ans)
    except:
        result.append(0)


def add_gensim_abstract(result, paper_info_1, paper_info_2):
    global gensim_abstract
    try:
        vec_1 = gensim_abstract.infer_vector(
            (paper_info_1['title'] + ' ' + paper_info_1['abstract']).split())
        vec_2 = gensim_abstract.infer_vector(
            (paper_info_2['title'] + ' ' + paper_info_2['abstract']).split())
        ans = cosVector(vec_1, vec_2)
        result.append(ans)
    except:
        result.append(0)


def compare_two_paper(paper_info_1, paper_info_2, author_rank):
    result = []
    the_author_name = paper_info_1['authors'][author_rank]['name']
    # 第一维度
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
    # 第二维度：是否来自相同的单位
    # todo : 这里是直接比较相等，其实名称还需要处理一下
    same_org = 0
    for author_2 in paper_info_2['authors']:
        if author_2['name'] == the_author_name:
            if ('org' in author_2) and ('org' in paper_info_1['authors'][author_rank]):
                if author_2['org'] == paper_info_1['authors'][author_rank]['org']:
                    same_org += 1
                    break
    if same_org >= 1:
        result.append(1)  # has same org
    else:
        result.append(0)
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
        result.append(abs(paper_info_1['year'] - paper_info_2['year']))
        # year gap between two papers=
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
    add_nltk_title(result, paper_info_1, paper_info_1)
    # 第七维度 abstract相似度
    add_nltk_abstract(result, paper_info_1, paper_info_1)
    # 第8维：gensim doc2vec
    add_gensim_title(result, paper_info_1, paper_info_2)
    # 第9维：gensim doc2vec
    add_gensim_abstract(result, paper_info_1, paper_info_2)
    return result


def replace_str(input):
    input = input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()
    return input.replace('yangjie', 'jieyang').replace('liubing', 'bingliu').replace('0008', '').replace('0002', '').replace('\xa0', '')


def get_nltk_result(docs):
    print("[train nltk]")
    global nltk_title, nltk_abstract
    whole_title = []
    whole_abstract = []
    for data in tqdm(docs):
        if 'title' in docs[data]:
            whole_title.append(fp.transform_sentence(docs[data]['title']))
        if 'abstract' in docs[data]:
            whole_abstract.append(fp.transform_sentence(docs[data]['abstract']))
    nltk_title = TextCollection(whole_title)
    nltk_abstract = TextCollection(whole_abstract)


def get_gensim_result():
    global gensim_title, gensim_abstract
    data_model_dir = 'data/track2/train/train_pub_gensim_model_'
    gensim_title = gensim.models.doc2vec.Doc2Vec.load(data_model_dir + "title.model")
    gensim_abstract = gensim.models.doc2vec.Doc2Vec.load(data_model_dir + "abstract.model")


if __name__ == "__main__":
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

    get_nltk_result(whole_author_profile_pub)
    get_gensim_result()

    result_dict = {}
    error_times = 0
    bst = xgb.Booster(model_file='xgb_1.model')
    print("Length ", len(cna_valid_unass_competition))
    for index, unass_data in tqdm(enumerate(cna_valid_unass_competition)):
        unass_paper_id = unass_data[:8]
        author_rank = int(unass_data[9:])
        unass_paper_info = cna_valid_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank]['name']

        cna_x = []
        id_list = []
        try:
            for same_name_author_id in whole_data_hash_by_name[replace_str(the_author_name)]:
                one_person_sim_list = []
                for paper_id in whole_data_hash_by_name[replace_str(the_author_name)][same_name_author_id]:
                    one_person_sim_list.append(compare_two_paper(unass_paper_info,
                                                                 whole_author_profile_pub[paper_id],
                                                                 author_rank))
                cna_x.append(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
                id_list.append(same_name_author_id)

            dtest=xgb.DMatrix(np.array(cna_x))
            ypred=bst.predict(dtest)
            predicted_author_id = id_list[np.argsort(ypred)[-1].item()]
            if predicted_author_id not in result_dict:
                result_dict[predicted_author_id] = []
            result_dict[predicted_author_id].append(unass_paper_id)
        except Exception as e:
            error_times += 1
            print('error_times: ', error_times)
            print(e)
            result_dict['laYWugfp'].append(unass_paper_id)
        # if index % 100 == 0:
        #     print('--- index: ' + str(index) + ' / ' + str(len(cna_valid_unass_competition)))

    with open('result.json', 'w') as w:
        w.write(json.dumps(result_dict))

