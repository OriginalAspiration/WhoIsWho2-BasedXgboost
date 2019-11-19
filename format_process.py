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


global whole_title
global whole_title_abstract


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
    global whole_title,whole_title_abstract
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


def nltk_tf_idf():
    global whole_title, whole_title_abstract
    # if_idf = if * idf 出现在越多的文档里的词越不重要
    # Question : 这个if_idf要不要按照作者分？
    corpus_title = TextCollection(whole_title)
    with open('data/track2/train/train_tf_idf.txt', 'wb') as model_tf_idf:
        pickle.dump(corpus_title, model_tf_idf)
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish title pickle ]")

    corpus_title_abstract = TextCollection(whole_title_abstract)
    with open('data/track2/train/train_abstract_tf_idf.txt', 'wb') as model_tf_idf2:
        pickle.dump(corpus_title_abstract, model_tf_idf2)
    print("[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]", "[Finish abstract pickle ]")


# gensim lsi


# doc2vec
def gensim_doc2vec():
    global whole_title, whole_title_abstract
    data_dir = "data/track2/train/gensim_doc2vec_"
    data = []

    title_x_train = []
    for i, doc in tqdm(enumerate(whole_title)):
        word_list = doc.split(' ')
        document = gensim.models.doc2vec.TaggedDocument(word_list, tags=[i])
        title_x_train.append(document)
    # 训练 Doc2Vec，并保存模型：
    # 实例化一个模型
    model_title = gensim.models.doc2vec.Doc2Vec(vector_size=256, window=10, min_count=5,
                                  workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
    model_title.build_vocab(title_x_train)
    print("开始训练...")
    # 训练模型
    model_title.train(title_x_train, total_examples=model_title.corpus_count, epochs=12)
    model_title.save(data_dir+"title.model")
    print("title model saved")

    abstract_x_train = []
    for i, doc in tqdm(enumerate(whole_title_abstract)):
        word_list = doc.split(' ')
        document = gensim.models.doc2vec.TaggedDocument(word_list, tags=[i])
        abstract_x_train.append(document)
    # 训练 Doc2Vec，并保存模型：
    # 实例化一个模型
    model_abstract = gensim.models.doc2vec.Doc2Vec(vector_size=256, window=10, min_count=5,
                                                workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
    model_abstract.build_vocab(abstract_x_train)
    print("开始训练...")
    # 训练模型
    model_abstract.train(abstract_x_train, total_examples=model_abstract.corpus_count, epochs=12)
    model_abstract.save(data_dir + "abstract.model")
    print("abstract model saved")

    test_text = whole_title[0].split()
    inferred_vector_dm = model_title.infer_vector(test_text)
    print(inferred_vector_dm)
    sims = model_title.docvecs.most_similar([inferred_vector_dm], topn=10)



if __name__ == '__main__':
    with open('data/track2/train/train_pub.json', 'r') as r:
        train_pub = json.load(r)
        print(len(train_pub))
        transform_pub(train_pub)
        # nltk_tf_idf(train_pub)
        gensim_doc2vec()
        # pre_tf_idf(train_pub)


