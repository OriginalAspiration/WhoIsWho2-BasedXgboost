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
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(doc)
    # print(word_list)
    lemmatize_output = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list])
    return lemmatize_output


def for_keywords_remove_stopword(doc):
    word_split = doc.split(" ")
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
    filtered_output = ' '.join([word for word in word_split if word not in stop_words])
    return filtered_output


def transform_sentence(doc):
    try:
        doc = doc.strip().replace('_', '').replace('-', '').replace('/sub', '').replace(' sub ', " ")
        doc = doc.replace("ABSTRACTS", '')
        doc = for_keywords_change_word_by_lemmatize(doc)
        doc = for_keywords_remove_stopword(doc)
    except:
        doc = ""
    return doc


def transform_pub(docs):
    for data in tqdm(docs):
        # 处理标题：词干化->去掉停用词->加入if-idf
        try:
            docs[data]['title'] = transform_sentence(docs[data]['title'])
        except ZeroDivisionError as e:
            docs[data]['title'] = ""
        # 处理关键词：词干化
        new_keywords_list = []
        try:
            if 'keywords' in docs[data]:
                new_keywords_list = [for_keywords_change_word_by_lemmatize(keywords) for keywords in cna_pub[data]['keywords']]
            else:
                new_keywords_list = []
        except ZeroDivisionError as e:
            new_keywords_list = []
        finally:
            docs[data]['keywords'] = new_keywords_list
        # 处理摘要: 词干化
        if 'abstract' in docs[data]:
            docs[data]['abstract'] = transform_sentence(docs[data]['abstract'])
        else:
            docs[data]['abstract'] = ""
    with open('data/track2/train/train_pub_alter.json', 'w', encoding='utf-8') as w:
        w.write(json.dumps(docs))
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")


if __name__ == '__main__':
    change_lemma = True
    with open('data/track2/train/train_pub.json', 'r') as r:
        train_pub = json.load(r)
        print("Length of train_pub", len(train_pub))
        if (not os.path.exists('data/track2/train/train_pub_alter.json')) or (change_lemma == True):
            transform_pub(train_pub)
        # pre_tf_idf(train_pub)


