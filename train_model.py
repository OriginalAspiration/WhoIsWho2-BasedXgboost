import nltk
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
import numpy as np
import multiprocessing
from multiprocessing import Pool

def cosVector(x, y):
    if len(x) != len(y):
        print('error input,x and y is not in the same space')
        return 0
    result1 = 1e-8
    result2 = 1e-8
    result3 = 1e-8
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
    for word in doc:
        if word not in word_dict:
            word_dict[word] = total_word
            total_word += 1


def get_doc_vector(word_dict, corpus, doc):
    dict_size = len(word_dict)
    doc_vector = [0 for i in range(dict_size)]
    tf = {}
    for word in doc:
        tf.setdefault(word, 0)
        tf[word] += 1
    
    for word in word_dict:
        if word in tf:
            word_index = word_dict[word]
            #print('word', word)
            ##print('tf', tf)
            #print('tf', tf[word])
            #print('corpus', corpus)
            #print('corpus', type(corpus))
            #print('corpus[word]', corpus[word])
            doc_vector[word_index] = tf[word]*1.0/ (corpus[word]+1.0)
    return doc_vector


def add_nltk_title(result, corpus, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    word_dict = {}
    
    pre_doc_vector(word_dict, docs[paper_id_1]['title'] + docs[paper_id_2]['title'])
    vector_1 = get_doc_vector(word_dict, corpus, docs[paper_id_1]['title'])
    vector_2 = get_doc_vector(word_dict, corpus, docs[paper_id_2]['title'])
    ans = cosVector(vector_1, vector_2)
    result[tup] = ans
    return ans



def add_nltk_abstract(result, corpus, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    
    word_dict = {}
    
    #TODO it may be some bug for why "" in abstart. They's id is WSNj1aYr
    if docs[paper_id_1]['abstract'] == "":
        docs[paper_id_1]['abstract'] = []
        print('paper_id_1', paper_id_1)
    if docs[paper_id_2]['abstract'] == "":
        docs[paper_id_2]['abstract'] = []
        print('paper_id_2', paper_id_2)

    pre_doc_vector(word_dict, docs[paper_id_1]['title']+ docs[paper_id_1]['abstract'] + docs[paper_id_2]['title'] + docs[paper_id_2]['abstract'])
    vector_1 = get_doc_vector(word_dict, corpus, docs[paper_id_1]['title'] + docs[paper_id_1]['abstract'])
    vector_2 = get_doc_vector(word_dict, corpus, docs[paper_id_2]['title'] + docs[paper_id_2]['abstract'])
    ans = cosVector(vector_1, vector_2)
    result[tup] = ans
    return ans



def nltk_calc_two_paper(corpus_title, corpus_abstract,
                        docs, paper_id_1, paper_id_2):
    '''if paper_id_1 not in docs or paper_id_2 not in docs:
        if paper_id_1 not in docs:
            print('xx paper_id_1', paper_id_1)
        if paper_id_2 not in docs:
            print('xx paper_id_2', paper_id_2)
        return 0, 0 , {(paper_id_1, paper_id_2): 0}, {(paper_id_1, paper_id_2): 0}
    else:'''
    result_title = {}
    result_abstract = {}
    ans1 = add_nltk_title(result_title, corpus_title, docs, paper_id_1, paper_id_2)
    ans2 = add_nltk_abstract(result_abstract, corpus_abstract, docs, paper_id_1, paper_id_2)
    return ans1, ans2, result_title, result_abstract

def nltk_idf(docs, data_dir):
    print("[train nltk]")
    corpus_title = {}
    corpus_abstract = {}

    for data in tqdm(docs):
        if 'title' in docs[data]:
            for x in docs[data]['title']:
                corpus_title.setdefault(x, 0)
                corpus_title[x] += 1
                corpus_abstract.setdefault(x, 0)
                corpus_abstract[x] += 1

        if 'abstract' in docs[data]:
            for x in docs[data]['abstract']:
                corpus_abstract.setdefault(x, 0)
                corpus_abstract[x] += 1
    # if_idf = if * idf 出现在越多的文档里的词越不重要
    
    with open(data_dir+'title.model', 'wb') as model_tf_idf:
        pickle.dump(corpus_title, model_tf_idf)

    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish title pickle ]")

    with open(data_dir+'abstract.model', 'wb') as model_tf_idf2:
        pickle.dump(corpus_abstract, model_tf_idf2)
        model_tf_idf2.close()
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish abstract pickle ]")


def f1(negative_example, existing_data_hash_by_name, corpus_title, corpus_abstract, pub, pool_id):
    print('pool_id', pool_id, 'begin')
    list_ans1 = []
    list_ans2 = []
    result_title = {}
    result_abstract = {}
    if pool_id == 7:
        x = tqdm(negative_example)
    else:
        x = negative_example
    for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in x:
        for paper_id in existing_data_hash_by_name[the_author_name][other_name_author_id]:
            ans1,ans2, r1, r2 = nltk_calc_two_paper(corpus_title,
                                    corpus_abstract,
                                    pub,
                                    unass_paper_id,
                                    paper_id)
            result_title.update(r1)
            result_abstract.update(r2)
            list_ans1.append(ans1)
            list_ans2.append(ans2)

    print("pool",pool_id , "is finish.", "[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")
    return list_ans1, list_ans2, result_title, result_abstract

def nltk_tf(pub, data_model_dir, data_result_dir, 
                        existing_data_hash_by_name,
                         positive_example,
                         negative_example):
    print("[begin nltk_tf]")
    
    with open(data_model_dir+'title.model', 'rb') as model_title:
        corpus_title = pickle.load(model_title)
        model_title.close()
    with open(data_model_dir+'abstract.model', 'rb') as model_abstract:
        corpus_abstract = pickle.load(model_abstract)
        model_abstract.close()

    result_title = {}
    result_abstract = {}

    if positive_example is not None:
        list_ans1 = []
        list_ans2 = []
        for unass_author_id, unass_paper_id, the_author_name, author_rank in tqdm(positive_example):
            for paper_id in existing_data_hash_by_name[the_author_name][unass_author_id]:
                #print('paper_id',paper_id, 'unass_paper_id', unass_paper_id)
                ans1,ans2, r1, r2 = nltk_calc_two_paper(corpus_title,
                                    corpus_abstract,
                                    pub,
                                    unass_paper_id,
                                    paper_id)
                result_title.update(r1)
                result_abstract.update(r2)
                list_ans1.append(ans1)
                list_ans2.append(ans2)
        print('mean ans1', np.mean(list_ans1))
        print('mean ans2', np.mean(list_ans2))
    
    
    
    num_pool = 8
    len_data = len(negative_example)
    print("Length of data", len_data)
    pool = Pool()
    step = len_data//num_pool

    id = 0
    sub_data = []
    jobs = []
    for data in negative_example:
        sub_data.append(data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name, corpus_title, corpus_abstract, pub, id)))
            id += 1
            sub_data = []

    if len(sub_data) >= 0:
        jobs.append(pool.apply_async(f1, args=(sub_data, existing_data_hash_by_name, corpus_title, corpus_abstract, pub, id)))
        id += 1
        sub_data = []

    pool.close()
    pool.join()

    list_ans1 = []
    list_ans2 = []
    for j in jobs:
        a1,a2, r1, r2 = j.get()
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
    
    print("[end nltk_tf]")
    return None


def train_nltk_model(file_name_unass_data, file_name_existing_data, file_name_pub, file_name_out,
                        existing_data_hash_by_name,
                         positive_example,
                         negative_example):
    with open(file_name_unass_data, 'r') as r:
        unass_data = json.load(r)

    with open(file_name_existing_data, 'r') as r:
        existing_data = json.load(r)

    with open(file_name_pub, 'r') as r:
        pub = json.load(r)

    data_model_dir = file_name_out + 'model_'
    # 训练模型
    nltk_idf(pub, data_model_dir)
    # 预处理出结果,get_train_data阶段可以直接用
    data_result_dir = file_name_out + 'result_'
    nltk_tf(pub, data_model_dir, data_result_dir,
            existing_data_hash_by_name,
            positive_example,
            negative_example)


# ========================================================
# Gensim
# doc2vec
# ========================================================

def gensim_train(docs, data_dir):
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
    vec1 = docs[paper_id_1]['doc2vec1']
    vec2 = docs[paper_id_1]['doc2vec1']
    # print(len(vec1), len(vec2))
    ans = cosVector(vec1, vec2)
    result[tup] = ans
    return ans



def add_gensim_abstract(result, docs, paper_id_1, paper_id_2):
    tup = (paper_id_1, paper_id_2)
    vec1 = docs[paper_id_1]['doc2vec2']
    vec2 = docs[paper_id_1]['doc2vec2']
    # print(len(vec1), len(vec2))
    ans = cosVector(vec1, vec2)
    result[tup] = ans
    return ans


def gensim_calc_two_paper(docs, paper_id_1, paper_id_2):
    '''if paper_id_1 not in docs or paper_id_2 not in docs:
        if paper_id_1 not in docs:
            print('xx paper_id_1', paper_id_1)
        if paper_id_2 not in docs:
            print('xx paper_id_2', paper_id_2)
        return 0, 0 , {(paper_id_1, paper_id_2): 0}, {(paper_id_1, paper_id_2): 0}
    else:'''
    result_title = {}
    result_abstract = {}
    ans1 = add_gensim_title(result_title, docs, paper_id_1, paper_id_2)
    ans2 = add_gensim_abstract(result_abstract, docs, paper_id_1, paper_id_2)
    return ans1, ans2, result_title, result_abstract

def f2(negative_example, existing_data_hash_by_name, pub, pool_id):
    print('pool_id', pool_id, 'begin')
    list_ans1 = []
    list_ans2 = []
    result_title = {}
    result_abstract = {}
    if pool_id == 7:
        x = tqdm(negative_example)
    else:
        x = negative_example
    for unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id in x:
        for paper_id in existing_data_hash_by_name[the_author_name][other_name_author_id]:
            ans1,ans2, r1, r2 = gensim_calc_two_paper(pub,
                                        unass_paper_id,
                                        paper_id)
            result_title.update(r1)
            result_abstract.update(r2)
            list_ans1.append(ans1)
            list_ans2.append(ans2)

    print("pool",pool_id , "is finish.", "[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")
    return list_ans1, list_ans2, result_title, result_abstract

def gensim_result(pub, data_model_dir, data_result_dir, existing_data_hash_by_name,
                         positive_example,
                         negative_example):

    model_title = gensim.models.doc2vec.Doc2Vec.load(data_model_dir + "title.model")
    model_abstract = gensim.models.doc2vec.Doc2Vec.load(data_model_dir + "abstract.model")
    print("[Gensim] each infer_vector")
    for data in tqdm(pub):
        if ('abstract' not in pub[data]) or (not pub[data]['abstract']):
            pub[data]['abstract'] = " "
        pub[data]['doc2vec1'] = model_title.infer_vector(pub[data]['title'].split())
        pub[data]['doc2vec2'] = model_abstract.infer_vector(
            (pub[data]['title'] + ' ' + pub[data]['abstract']).split())

    print("[Gensim] calc two sim")

    result_title = {}
    result_abstract = {}

    if positive_example is not None:
        list_ans1 = []
        list_ans2 = []
        for unass_author_id, unass_paper_id, the_author_name, author_rank in tqdm(positive_example):
            for paper_id in existing_data_hash_by_name[the_author_name][unass_author_id]:
                #print('paper_id',paper_id, 'unass_paper_id', unass_paper_id)
                ans1,ans2, r1, r2 = gensim_calc_two_paper(pub,
                                    unass_paper_id,
                                    paper_id)
                result_title.update(r1)
                result_abstract.update(r2)
                list_ans1.append(ans1)
                list_ans2.append(ans2)
        print('mean ans1', np.mean(list_ans1))
        print('mean ans2', np.mean(list_ans2))
    
    
    num_pool = 8
    len_data = len(negative_example)
    print("Length of data", len_data)
    pool = Pool()
    step = len_data//num_pool

    id = 0
    sub_data = []
    jobs = []
    for data in negative_example:
        sub_data.append(data)
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(f2, args=(sub_data, existing_data_hash_by_name, pub, id)))
            id += 1
            sub_data = []
    if len(sub_data) > 0:
        jobs.append(pool.apply_async(f2, args=(sub_data, existing_data_hash_by_name, pub, id)))
        id += 1
        sub_data = []
    
    pool.close()
    pool.join()

    list_ans1 = []
    list_ans2 = []
    for j in jobs:
        a1,a2, r1, r2 = j.get()
        list_ans1 += a1
        list_ans2 += a2
        result_title.update(r1)
        result_abstract.update(r2)
        
    print('mean ans1', np.mean(list_ans1))
    print('mean ans2', np.mean(list_ans2))

    print(len(result_title), type(result_title))
    #print(type(json.dumps(result_title)))
    print(len(result_abstract))
    with open(data_result_dir+'title.json', 'wb') as fresult_title:
        pickle.dump(result_title, fresult_title)

    with open(data_result_dir+'abstract.json', 'wb') as fresult_abstract:
        pickle.dump(result_abstract, fresult_abstract)

    return


def train_gensim_model(file_name_unass_data, file_name_existing_data, file_name_pub, file_name_out,
                        existing_data_hash_by_name,
                         positive_example,
                         negative_example):
    # if os.path.exists(file_name_out+'map_title.model'):
    #     return
    with open(file_name_unass_data, 'r') as r:
        unass_data = json.load(r)

    with open(file_name_existing_data, 'r') as r:
        existing_data = json.load(r)

    with open(file_name_pub, 'r') as r:
        pub = json.load(r)

    data_model_dir = file_name_out + 'model_'
    # 训练模型
    gensim_train(pub, data_model_dir)
    # 预处理出结果,get_train_data阶段可以直接用
    data_result_dir = file_name_out + 'result_'
    gensim_result(pub, data_model_dir, data_result_dir, existing_data_hash_by_name,
                         positive_example,
                         negative_example)
    return


if __name__ == "__main__":
    train_nltk = True
    train_gensim = False

    file_name_unass_data = 'data/track2/train/train_unass_data.json'
    file_name_existing_data = 'data/track2/train/train_existing_data.json'
    file_name_pub = 'data/track2/train/train_pub_alter.json'
    random.seed(2333)

    if train_nltk or train_gensim:
        with open(file_name_unass_data, 'r') as r:
            unass_data = json.load(r)

        with open(file_name_existing_data, 'r') as r:
            existing_data = json.load(r)

        with open(file_name_pub, 'r') as r:
            pub = json.load(r)
        
        positive_example = []
        negative_example = []
        existing_data_hash_by_name = {}
        for person_id in existing_data:
            real_name = existing_data[person_id]['name']
            replaced_real_name = replace_str(real_name)
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
            the_author_name = replace_str(unass_paper_info['authors'][author_rank]['name'])
            # 正样本：同名作者
            positive_example.append([unass_author_id, unass_paper_id, the_author_name, author_rank])
            

            # 随机负样本： 不同id作者
            if len(existing_data_hash_by_name[the_author_name]) > 1:
                diff_name_author_id = list(existing_data_hash_by_name[the_author_name])
                diff_name_author_id.remove(unass_author_id)
                # print(diff_name_author_id)
                
                #other_name_author_id = random.choice(diff_name_author_id)
                #negative_example.append([unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id])

                for other_name_author_id in diff_name_author_id:
                    negative_example.append([unass_author_id, unass_paper_id, the_author_name, author_rank, other_name_author_id])
        
        with open('data/track2/train/training_data.pkl', 'wb') as file:
            pickle.dump([existing_data_hash_by_name,positive_example,negative_example], file)

    
    if train_nltk:
        out_file_name = 'data/track2/train/train_pub_nltk_'
        train_nltk_model(file_name_unass_data,
                         file_name_existing_data,
                         file_name_pub,
                         out_file_name,
                         existing_data_hash_by_name,
                         positive_example,
                         negative_example)
    


    file_name_pub = 'data/track2/train/train_pub.json'
    if train_gensim:
        out_file_name = 'data/track2/train/train_pub_gensim_'
        train_gensim_model(file_name_unass_data,
                           file_name_existing_data,
                           file_name_pub,
                           out_file_name,
                        existing_data_hash_by_name,
                        positive_example,
                        negative_example)
