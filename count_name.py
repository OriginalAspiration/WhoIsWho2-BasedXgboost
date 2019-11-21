import time
import json
import pickle

import numpy as np
import xgboost as xgb


def replace_str(input):
    input = input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()
    return input.replace('jie yang', 'jieyang').replace('yangjie', 'jieyang').replace('liubing', 'bingliu').replace('0008', '').replace('0002', '')


if __name__ == "__main__":
    with open('data/track2/cna_data/cna_valid_unass_competition.json', 'r') as r:
        cna_valid_unass_competition = json.load(r)
    with open('data/track2/cna_data/cna_valid_pub.json', 'r') as r:
        cna_valid_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile.json', 'r') as r:
        whole_author_profile = json.load(r)

    whole_data_hash_by_name = {}
    for person_id in whole_author_profile:
        real_name = whole_author_profile[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in whole_data_hash_by_name:
            whole_data_hash_by_name[replaced_real_name] = {}
        whole_data_hash_by_name[replaced_real_name][person_id] = whole_author_profile[person_id]['papers']

    count_existing = 0
    count_total = 0
    for index, unass_data in enumerate(cna_valid_unass_competition):
        unass_paper_id = unass_data[:8]
        author_rank = int(unass_data[9:])
        unass_paper_info = cna_valid_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank]['name']
        if not replace_str(the_author_name) in whole_data_hash_by_name:
            count_existing += 1
            print('---' + the_author_name + '---' + replace_str(the_author_name) + '---')
