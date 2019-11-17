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


def transform_pub(cna_pub):
    whole_title = []
    total_word_title = 0
    whole_title_abstract = []
    total_word_abstract = 0
    for data in tqdm(cna_pub):
        # 处理标题：词干化->去掉停用词->加入if-idf
        try:
            cna_pub[data]['title'] = transform_sentence(cna_pub[data]['title'])
        except ZeroDivisionError as e:
            cna_pub[data]['title'] = ""
        finally:
            whole_title.append(cna_pub[data]['title'])
            # print("\ttitle:", cna_pub[data]['title'])

        # 处理关键词：词干化
        new_keywords_list = []
        try:
            if 'keywords' in cna_pub[data]:
                new_keywords_list = [for_keywords_change_word_by_lemmatize(keywords) for keywords in cna_pub[data]['keywords']]
            else:
                new_keywords_list = []
        except ZeroDivisionError as e:
            new_keywords_list = []
        finally:
            cna_pub[data]['keywords'] = new_keywords_list
            # print("\tkeywords:", cna_pub[data]['keywords'])

        # 处理摘要: 词干化
        if 'abstract' in cna_pub[data]:
            cna_pub[data]['abstract'] = transform_sentence(cna_pub[data]['abstract'])
        else:
            cna_pub[data]['abstract'] = ""
        whole_title_abstract.append(cna_pub[data]['title'] + cna_pub[data]['abstract'])
        # outf.write("passage : " + str(total) + "\n")
        # outf.write("\ttitle:" + cna_pub[data]['title'] + "\n")
        # outf.write("\tkeywords:" + (" ".join(cna_pub[data]['keywords'])) + "\n")
        # total += 1
    with open('data/track2/train/train_pub_alter.json', 'w', encoding='utf-8') as w:
        w.write(json.dumps(cna_pub))
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish lemma ]")
    # if_idf = if * idf 出现在越多的文档里的词越不重要
    # Question : 这个if_idf要不要按照作者分？
    corpus_title = TextCollection(whole_title)
    with open('data/track2/train/train_tf_idf.txt', 'wb') as model_tf_idf:
        pickle.dump(corpus_title, model_tf_idf)
    # eps = 0.0       # 单词频率精度
    # word_dict = {}
    # for sentence in tqdm(whole_title):
    #     for word in sentence.split():
    #         if (corpus_title.idf(word) > eps) and (word not in word_dict):
    #             word_dict[word] = total_word_title
    #             total_word_title += 1
    #
    # with open('data/track2/train/train_word_dict.txt', 'wb') as word_file:
    #     pickle.dump(word_dict, word_file)
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish title pickle ]")
    corpus_title_abstract = TextCollection(whole_title_abstract)
    with open('data/track2/train/train_abstract_tf_idf.txt', 'wb') as model_tf_idf2:
        pickle.dump(corpus_title_abstract, model_tf_idf2)
    # word_dict_abstract = {}
    # ftotal = [0, 0, 0, 0, 0]
    # feps = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    # for sentence in tqdm(whole_title_abstract):
    #     for word in sentence.split():
    #         if (corpus_title_abstract.idf(word) > eps) and (word not in word_dict_abstract):
    #             now = corpus_title_abstract.idf(word)
    #             for i in range(5):
    #                 if now >= feps[i]:
    #                     ftotal[i] += 1
    #             word_dict_abstract[word] = total_word_abstract
    #             total_word_abstract += 1
    # with open('data/track2/train/train_abstract_word_dict.txt', 'wb') as word_file2:
    #     pickle.dump(word_dict_abstract, word_file2)
    # print("Length of Word_abstract:", len(word_dict_abstract))
    # print("Length of Word_title:", len(word_dict))
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish abstract pickle ]")
    # for i in range(5):
    #     print("Eps ", feps[i], 'Cnt ', ftotal[i])
    #
    # doc_1 = whole_title[1]
    # doc_2 = whole_title[2]
    # dict_size = len(word_dict)
    # doc_vector_1 = [0 for i in range(dict_size)]
    # for word in doc_1.split():
    #     word_index = word_dict[word]
    #     doc_vector_1[word_index] = corpus_title.tf_idf(word, doc_1)
    # doc_vector_2 = [0 for i in range(dict_size)]
    # for word in doc_2.split():
    #     word_index = word_dict[word]
    #     doc_vector_2[word_index] = corpus_title.tf_idf(word, doc_2)
    # cosVector(doc_vector_1, doc_vector_2)


if __name__ == '__main__':
    with open('data/track2/train/train_pub.json', 'r') as r:
        train_pub = json.load(r)
        print(len(train_pub))
        transform_pub(train_pub)
        # pre_tf_idf(train_pub)


