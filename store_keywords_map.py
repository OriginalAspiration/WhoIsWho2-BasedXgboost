import re
import time
import json
import pickle

import torch
import numpy as np
import xgboost as xgb
from tqdm import tqdm

import nltk.stem.porter as pt
pt_stemmer = pt.PorterStemmer()


def process_keyword(formated_keyword):
    w = ''.join([pt_stemmer.stem( re.sub(r'[^a-zA-Z]', '', word ) ) for word in formated_keyword if len(word) > 0])
    return w # type: str


if __name__ == "__main__":
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)

    k2k_edge_times = {}
    for pub_id in tqdm(train_pub):
        if len(train_pub[pub_id]['keywords']) > 10 or len(train_pub[pub_id]['keywords']) < 2:
            continue
        for keyword_1 in train_pub[pub_id]['keywords']:
            keyword_1p = process_keyword(keyword_1)
            if keyword_1p not in k2k_edge_times:
                k2k_edge_times[keyword_1p] = {}
            for keyword_2 in train_pub[pub_id]['keywords']:
                keyword_2p = process_keyword(keyword_2)
                if keyword_1p == keyword_2p:
                    continue
                if keyword_2p not in k2k_edge_times[keyword_1p]:
                    k2k_edge_times[keyword_1p][keyword_2p] = 0
                k2k_edge_times[keyword_1p][keyword_2p] += 1

    print('----- clean k2k_edge_times -----')
    key_word_num = len(k2k_edge_times)
    print('key_word_num', key_word_num)
    stop_words = []
    for keyword_1 in k2k_edge_times:
        if len(k2k_edge_times[keyword_1].keys()) > key_word_num // 100:
            stop_words.append(keyword_1)
    stop_word_num = len(stop_words)
    print('stop_word_num', stop_word_num)
    stop_words = set(stop_words)

    k2k_edges = {}
    n = 0
    for keyword_1 in k2k_edge_times:
        if keyword_1 in stop_words:
            continue
        if keyword_1 not in k2k_edges:
            k2k_edges[keyword_1] = []
        for keyword_2 in k2k_edge_times[keyword_1]:
            if keyword_2 in stop_words:
                continue
            if k2k_edge_times[keyword_1][keyword_2] > len(train_pub) // 10000:
                k2k_edges[keyword_1].append(keyword_2)
                n += 1
        k2k_edges[keyword_1] = set(k2k_edges[keyword_1])
    print('----- Strong keywords link num:', n)

    with open('data/track2/train/keywords_map.pkl', 'wb') as wb:
        pickle.dump(k2k_edges, wb)


