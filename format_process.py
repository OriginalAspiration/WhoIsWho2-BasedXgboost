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
from gensim import corpora, models, similarities
import os
import multiprocessing
from multiprocessing import Pool


lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# 词性还原
def for_keywords_change_word_by_lemmatize(doc):
    word_list = nltk.word_tokenize(doc)
    return [lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w)) for w in word_list]


def for_keywords_remove_stopword(doc):
    word_split = doc
    valid_word = []
    for word in word_split:
        word = word.strip(" ").strip(string.digits)
        if word != "":
            valid_word.append(word)
    word_split = valid_word
    stop_words = set(stopwords.words('english'))
    # add punctuations
    punctuations = list(string.punctuation)
    [stop_words.add(punc) for punc in punctuations]
    # remove null
    stop_words.add("null")

    return [word for word in word_split if word not in stop_words]


def transform_sentence(doc):
    if doc is None:
        return ""
    #print('doc', doc)
    doc = doc.strip().replace('_', '').replace(
        '-', '').replace('/sub', '').replace(' sub ', " ")
    doc = doc.replace("ABSTRACTS", '')
    doc = for_keywords_change_word_by_lemmatize(doc)
    doc = for_keywords_remove_stopword(doc)
    #print('doc', doc)
    return doc


def transform_pub(docs, pool_id=0):
    print('pool_id', pool_id, 'begin')
    if pool_id == 0:
        x = tqdm(docs)
    else:
        x = docs
    new_docs = docs
    for data in x:
        try:
            # 处理标题：词干化->去掉停用词->加入if-idf
            if 'title' in docs[data]:
                new_docs[data]['title'] = transform_sentence(docs[data]['title'])
            else:
                new_docs[data]['title'] = []

            # 处理关键词：词干化
            new_keywords_list = []

            if 'keywords' in docs[data]:
                new_keywords_list = [for_keywords_change_word_by_lemmatize(
                    keywords) for keywords in docs[data]['keywords']]
            else:
                new_keywords_list = []
            new_docs[data]['keywords'] = new_keywords_list

            # 处理摘要: 词干化
            if 'abstract' in docs[data]:
                new_docs[data]['abstract'] = transform_sentence(docs[data]['abstract'])
            else:
                new_docs[data]['abstract'] = []
        except:
            print('data', data, 'is warn')
    print("pool", pool_id, "is finish.", "[", time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")
    return new_docs


def multi_process_format_data(pub, num_pool=8):
    print('--- multi_process_format_data START ---')
    len_data = len(pub)
    print("Length of data", len_data)
    pool = Pool()  # train_pub
    step = len_data // num_pool
    id = 0
    sub_data = {}
    jobs = []
    for data in pub:
        sub_data[data] = pub[data]
        if len(sub_data) >= step:
            jobs.append(pool.apply_async(transform_pub, args=(sub_data, id)))
            id += 1
            sub_data = {}
    if len(sub_data) > 0:
        jobs.append(pool.apply_async(transform_pub, args=(sub_data, id)))
        id += 1
        sub_data = {}

    pool.close()
    pool.join()

    results = {}
    for j in jobs:
        x = j.get()
        results.update(x)

    return results


if __name__ == '__main__':
    with open('data/track2/train/train_pub.json', 'r') as r:
        train_pub = json.load(r)

    new_pub = multi_process_format_data(train_pub)

    with open(save_path, 'w', encoding='utf-8') as w:
        w.write(json.dumps(results))
