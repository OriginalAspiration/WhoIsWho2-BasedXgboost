import time
import json
import pickle

import numpy as np
import xgboost as xgb


def replace_str(input):
    return input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()


if __name__ == "__main__":
    with open('data/track2/cna_data/whole_author_profile.json', 'r') as r:
        whole_author_profile = json.load(r)

    whole_data_hash_by_name = {}
    for person_id in whole_author_profile:
        real_name = whole_author_profile[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in whole_data_hash_by_name:
            whole_data_hash_by_name[replaced_real_name] = real_name

    with open('whole_data_hash_by_name.json', 'w') as w:
        w.write(json.dumps(whole_data_hash_by_name))
