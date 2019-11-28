import re
import time
import json
import pickle

import torch
import numpy as np
import xgboost as xgb
from tqdm import tqdm

def fix_name(s):
    s = s.lower().strip()
    x = re.split(r'[ \.\-\_]', s)
    set_x = set()
    for a in x:
        if len(a) > 0:
            set_x.add(a)
    x = list(set_x)
    x.sort()
    s = ''.join(x)
    return s


if __name__ == "__main__":
    with open('data/track2/train/train_pub_alter.json', 'r') as r:
        train_pub = json.load(r)

    n2n_edge_times = {}
    for pub_id in tqdm(train_pub):
        if len(train_pub[pub_id]['authors']) > 10 or len(train_pub[pub_id]['authors']) < 2:
            continue
        for author_1 in train_pub[pub_id]['authors']:
            name_1 = fix_name(author_1['name'])
            if name_1 not in n2n_edge_times:
                n2n_edge_times[name_1] = {}
            for author_2 in train_pub[pub_id]['authors']:
                name_2 = fix_name(author_2['name'])
                if name_1 == name_2:
                    continue
                if name_2 not in n2n_edge_times[name_1]:
                    n2n_edge_times[name_1][name_2] = 0
                n2n_edge_times[name_1][name_2] += 1

    print('----- clean n2n_edge_times -----')
    name_num = len(n2n_edge_times)
    print('name_num', name_num)
    stop_name = []
    for name in n2n_edge_times:
        if len(n2n_edge_times[name].keys()) > 100:
            stop_name.append(name)
    stop_name_num = len(stop_name)
    print('stop_name_num', stop_name_num)
    stop_name = set(stop_name)

    n2n_edges = {}
    n = 0
    for name_1 in n2n_edge_times:
        if name_1 in stop_name:
            continue
        if name_1 not in n2n_edges:
            n2n_edges[name_1] = []
        for name_2 in n2n_edge_times[name_1]:
            if name_2 in stop_name:
                continue
            if n2n_edge_times[name_1][name_2] > 5:
                n2n_edges[name_1].append(name_2)
                n += 1
        n2n_edges[name_1] = set(n2n_edges[name_1])
    print('----- Strong name link num:', n)

    with open('data/track2/train/name_map.pkl', 'wb') as wb:
        pickle.dump(n2n_edges, wb)


