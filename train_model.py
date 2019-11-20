import nltk
from nltk.text import TextCollection
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
import json
import pickle
import time
from tqdm import tqdm
import gensim
import random
from gensim import corpora, models, similarities
import os


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


def replace_str(input):
    return input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()


# ====================================
# nltk
# ====================================
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


def add_nltk_title(result, corpus, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    try:
        word_dict = {}
        pre_doc_vector(word_dict, docs[paper_id_1]['title'] + ' ' + docs[paper_id_2]['title'])
        vector_1 = get_doc_vector(word_dict, corpus, docs[paper_id_1]['title'])
        vector_2 = get_doc_vector(word_dict, corpus, docs[paper_id_2]['title'])
        ans = cosVector(vector_1, vector_2)
        result[tup] = ans
    except:
        result[tup] = 0


def add_nltk_abstract(result, corpus, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    try:
        word_dict = {}
        pre_doc_vector(word_dict, docs[paper_id_1]['title'] + ' ' + docs[paper_id_1]['abstract']
                        + ' ' + docs[paper_id_2]['title'] + ' ' + docs[paper_id_2]['abstract'])
        vector_1 = get_doc_vector(word_dict, corpus, docs[paper_id_1]['title'] + ' ' + docs[paper_id_1]['abstract'])
        vector_2 = get_doc_vector(word_dict, corpus, docs[paper_id_2]['title'] + ' ' + docs[paper_id_2]['abstract'])
        ans = cosVector(vector_1, vector_2)
        result[tup] = ans
        # print("Cos:",ans)
    except:
        result[tup] = 0


def nltk_calc_two_paper(corpus_title, corpus_abstract,
                        result_title, result_abstract,
                        docs, paper_id_1, paper_id_2):
    add_nltk_title(result_title, corpus_title, docs, paper_id_1, paper_id_2)
    add_nltk_abstract(result_abstract, corpus_abstract, docs, paper_id_1, paper_id_2)


def nltk_idf(docs, data_dir):
    if os.path.exists(data_dir+'title.model'):
        return
    print("[train nltk]")
    whole_title = []
    whole_abstract = []
    for data in tqdm(docs):
        if 'title' in docs[data]:
            whole_title.append(docs[data]['title'])
        if 'abstract' in docs[data]:
            whole_abstract.append(docs[data]['abstract'])
    # if_idf = if * idf 出现在越多的文档里的词越不重要
    # Question : 这个if_idf要不要按照作者分？
    corpus_title = TextCollection(whole_title)
    with open(data_dir+'title.model', 'wb') as model_tf_idf:
        pickle.dump(corpus_title, model_tf_idf)
        model_tf_idf.close()
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish title pickle ]")

    corpus_abstract = TextCollection(whole_abstract)
    with open(data_dir+'abstract.model', 'wb') as model_tf_idf2:
        pickle.dump(corpus_abstract, model_tf_idf2)
        model_tf_idf2.close()
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish abstract pickle ]")


def nltk_tf(unass_data, existing_data, pub, data_model_dir, data_result_dir):
    if os.path.exists(data_result_dir+'title.json'):
        return
    existing_data_hash_by_name = {}
    for person_id in existing_data:
        real_name = existing_data[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in existing_data_hash_by_name:
            existing_data_hash_by_name[replaced_real_name] = {}
        existing_data_hash_by_name[replaced_real_name][person_id] = existing_data[person_id]['papers']
    with open(data_model_dir+'title.model', 'rb') as model_title:
        corpus_title = pickle.load(model_title)
        model_title.close()
    with open(data_model_dir+'abstract.model', 'rb') as model_abstract:
        corpus_abstract = pickle.load(model_abstract)
        model_abstract.close()

    random.seed(2333)
    total = 0
    result_title = {}
    result_abstract = {}
    for unass_data in tqdm(unass_data):
        unass_paper_id = unass_data[0][:8]
        author_rank = int(unass_data[0][9:])
        unass_author_id = unass_data[1]
        unass_paper_info = pub[unass_paper_id]
        the_author_name = replace_str(unass_paper_info['authors'][author_rank]['name'])
        # 正样本：同名作者
        # print("\n", existing_data_hash_by_name[the_author_name])
        for same_name_author_id in existing_data_hash_by_name[the_author_name]:
            if same_name_author_id == unass_author_id:
                one_person_sim_list = []
                for paper_id in existing_data_hash_by_name[the_author_name][same_name_author_id]:
                    one_person_sim_list.append(
                        nltk_calc_two_paper(corpus_title,
                                            corpus_abstract,
                                            result_title,
                                            result_abstract,
                                            pub,
                                            unass_paper_id,
                                            paper_id))
                break

        # 随机负样本： 不同id作者
        if len(existing_data_hash_by_name[the_author_name]) > 1:
            diff_name_author_id = list(existing_data_hash_by_name[the_author_name])
            diff_name_author_id.remove(unass_author_id)
            # print(diff_name_author_id)
            same_name_author_id = random.choice(diff_name_author_id)
            one_person_sim_list = []
            for paper_id in existing_data_hash_by_name[the_author_name][same_name_author_id]:
                one_person_sim_list.append(
                    nltk_calc_two_paper(corpus_title,
                                        corpus_abstract,
                                        result_title,
                                        result_abstract,
                                        pub,
                                        unass_paper_id,
                                        paper_id))
        total += 1
        if total == 100:
            break
    print(len(result_title))
    print(len(result_abstract))
    with open(data_result_dir + 'title.res', 'wb') as fresult_title:
        pickle.dump(result_title, fresult_title)
        fresult_title.close()
    with open(data_result_dir + 'abstract.res', 'wb') as fresult_abstract:
        pickle.dump(result_abstract, fresult_abstract)
        fresult_title.close()
    return


def train_nltk_model(file_name_unass_data, file_name_existing_data, file_name_pub, file_name_out):
    if os.path.exists(file_name_out+'result_title.model'):
        return
    with open(file_name_unass_data, 'r') as r:
        unass_data = json.load(r)
        r.close()
    with open(file_name_existing_data, 'r') as r:
        existing_data = json.load(r)
        r.close()
    with open(file_name_pub, 'r') as r:
        pub = json.load(r)
        r.close()
    data_model_dir = file_name_out + 'model_'
    # 训练模型
    nltk_idf(pub, data_model_dir)
    # 预处理出结果,get_train_data阶段可以直接用
    data_result_dir = file_name_out + 'result_'
    nltk_tf(unass_data, existing_data, pub, data_model_dir, data_result_dir)


# ========================================================
# Gensim
# doc2vec
# ========================================================

def gensim_train(docs, data_dir):
    if os.path.exists(data_dir+'title.model'):
        return
    print("[Gensim] train gensim")
    whole_title = []
    whole_abstract = []
    for data in tqdm(docs):
        if 'title' in docs[data]:
            whole_title.append(docs[data]['title'])
        if ('abstract' not in docs[data]) or (not docs[data]['abstract']):
            docs[data]['abstract'] = " "
        whole_abstract.append(docs[data]['abstract'])

    # 训练 title
    title_x_train = []
    for i, doc in tqdm(enumerate(whole_title)):
        word_list = doc.split(' ')
        if len(word_list) > 0:
            document = gensim.models.doc2vec.TaggedDocument(word_list, tags=[i])
            title_x_train.append(document)
    model_title = gensim.models.doc2vec.Doc2Vec(vector_size=256, window=10, min_count=5,
                                  workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
    model_title.build_vocab(title_x_train)
    print("[Gensim] 开始训练title...")
    model_title.train(title_x_train, total_examples=model_title.corpus_count, epochs=12)
    model_title.save(data_dir+"title.model")
    print("[Gensim] title model saved")

    # 训练abstract
    abstract_x_train = []
    for i, doc in tqdm(enumerate(whole_abstract)):
        word_list = doc.split(' ')
        if len(word_list) > 0:
            document = gensim.models.doc2vec.TaggedDocument(word_list, tags=[i])
            abstract_x_train.append(document)
    model_abstract = gensim.models.doc2vec.Doc2Vec(vector_size=256, window=10, min_count=5,
                                                workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
    model_abstract.build_vocab(abstract_x_train)
    print("[Gensim] 开始训练abstract...")
    # 训练模型
    model_abstract.train(abstract_x_train, total_examples=model_abstract.corpus_count, epochs=12)
    model_abstract.save(data_dir + "abstract.model")
    print("[Gensim] abstract model saved")


def add_gensim_title(result, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    try:
        vec1 = docs[paper_id_1]['doc2vec1']
        vec2 = docs[paper_id_1]['doc2vec1']
        # print(len(vec1), len(vec2))
        ans = cosVector(vec1, vec2)
        result[tup] = ans
    except:
        result[tup] = 0


def add_gensim_abstract(result, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    try:
        vec1 = docs[paper_id_1]['doc2vec2']
        vec2 = docs[paper_id_1]['doc2vec2']
        # print(len(vec1), len(vec2))
        ans = cosVector(vec1, vec2)
        result[tup] = ans
    except:
        result[tup] = 0


def gensim_calc_two_paper(result_title, result_abstract,
                          docs, paper_id_1, paper_id_2):
    add_gensim_title(result_title, docs, paper_id_1, paper_id_2)
    add_gensim_abstract(result_abstract, docs, paper_id_1, paper_id_2)


def gensim_result(unass_data, existing_data, pub, data_model_dir, data_result_dir):
    if os.path.exists(data_result_dir+'title.json'):
        return
    existing_data_hash_by_name = {}
    for person_id in existing_data:
        real_name = existing_data[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in existing_data_hash_by_name:
            existing_data_hash_by_name[replaced_real_name] = {}
        existing_data_hash_by_name[replaced_real_name][person_id] = existing_data[person_id]['papers']
    model_title = gensim.models.doc2vec.Doc2Vec.load(data_model_dir + "title.model")
    model_abstract = gensim.models.doc2vec.Doc2Vec.load(data_model_dir + "abstract.model")
    print("[Gensim] each infer_vector")
    for data in tqdm(pub):
        if ('abstract' not in pub[data]) or (not pub[data]['abstract']):
            pub[data]['abstract'] = " "
        pub[data]['doc2vec1'] = model_title.infer_vector(pub[data]['title'].split())
        pub[data]['doc2vec2'] = model_abstract.infer_vector(
            (pub[data]['title'] + ' ' + pub[data]['abstract']).split())

    random.seed(2333)
    result_title = {}
    result_abstract = {}
    total = 0
    print("[Gensim] calc two sim")
    for unass_data in tqdm(unass_data):
        unass_paper_id = unass_data[0][:8]
        author_rank = int(unass_data[0][9:])
        unass_author_id = unass_data[1]
        unass_paper_info = pub[unass_paper_id]
        the_author_name = replace_str(unass_paper_info['authors'][author_rank]['name'])
        # 正样本：同名作者
        # print("\n", existing_data_hash_by_name[the_author_name])
        for same_name_author_id in existing_data_hash_by_name[the_author_name]:
            if same_name_author_id == unass_author_id:
                one_person_sim_list = []
                for paper_id in existing_data_hash_by_name[the_author_name][same_name_author_id]:
                    one_person_sim_list.append(
                        gensim_calc_two_paper(result_title,
                                              result_abstract,
                                              pub,
                                              unass_paper_id,
                                              paper_id))
                break

        # 随机负样本： 不同id作者
        if len(existing_data_hash_by_name[the_author_name]) > 1:
            diff_name_author_id = list(existing_data_hash_by_name[the_author_name])
            diff_name_author_id.remove(unass_author_id)
            # print(diff_name_author_id)
            same_name_author_id = random.choice(diff_name_author_id)
            one_person_sim_list = []
            for paper_id in existing_data_hash_by_name[the_author_name][same_name_author_id]:
                one_person_sim_list.append(
                    gensim_calc_two_paper(result_title,
                                          result_abstract,
                                          pub,
                                          unass_paper_id,
                                          paper_id))
        total += 1
        if total == 100:
            break
    print(len(result_title))
    with open(data_result_dir+'title.json', 'wb') as fresult_title:
        pickle.dump(result_title, fresult_title)
        fresult_title.close()
    with open(data_result_dir+'abstract.json', 'wb') as fresult_abstract:
        pickle.dump(result_abstract, fresult_abstract)
        fresult_title.close()
    return


def train_gensim_model(file_name_unass_data, file_name_existing_data, file_name_pub, file_name_out):
    # if os.path.exists(file_name_out+'map_title.model'):
    #     return
    with open(file_name_unass_data, 'r') as r:
        unass_data = json.load(r)
        r.close()
    with open(file_name_existing_data, 'r') as r:
        existing_data = json.load(r)
        r.close()
    with open(file_name_pub, 'r') as r:
        pub = json.load(r)
        r.close()
    data_model_dir = file_name_out + 'model_'
    # 训练模型
    gensim_train(pub, data_model_dir)
    # 预处理出结果,get_train_data阶段可以直接用
    data_result_dir = file_name_out + 'result_'
    gensim_result(unass_data, existing_data, pub, data_model_dir, data_result_dir)
    return


if __name__ == "__main__":
    file_name_unass_data = 'data/track2/train/train_unass_data.json'
    file_name_existing_data = 'data/track2/train/train_existing_data.json'
    file_name_pub = 'data/track2/train/train_pub_alter.json'
    train_nltk = True
    if train_nltk:
        out_file_name = 'data/track2/train/train_pub_nltk_'
        train_nltk_model(file_name_unass_data,
                         file_name_existing_data,
                         file_name_pub,
                         out_file_name)
    file_name_unass_data = 'data/track2/train/train_unass_data.json'
    file_name_existing_data = 'data/track2/train/train_existing_data.json'
    file_name_pub = 'data/track2/train/train_pub.json'
    train_gensim = True
    if train_gensim:
        out_file_name = 'data/track2/train/train_pub_gensim_'
        train_gensim_model(file_name_unass_data,
                           file_name_existing_data,
                           file_name_pub,
                           out_file_name)
