import time
import json
import pickle

import numpy as np


def compare_two_paper(paper_info_1, paper_info_2, author_rank):
    try:
        result = []
        the_author_name = paper_info_1['authors'][author_rank-1]['name']

        same_author = 0
        for author_1 in paper_info_1['authors']:
            for author_2 in paper_info_2['authors']:
                if author_1['name'] == author_2['name']:
                    same_author += 1
                    break
            if same_author >= 2:
                break
        if same_author >= 2:
            result.append(1) # has same collaborator
        else:
            result.append(0)

        same_org = 0
        for author_2 in paper_info_2['authors']:
            if author_2['name'] == the_author_name:
                if author_2['org'] == paper_info_1['authors'][author_rank-1]['org']:
                    same_org += 1
                    break
        if same_org >= 1:
            result.append(1) # has same org
        else:
            result.append(0)

        if paper_info_1['venue'] == paper_info_2['venue']:
            result.append(1) # has same venue
        else:
            result.append(0)

        result.append(abs(paper_info_1['year'] - paper_info_2['year'])) # year gap between two papers

        same_keyword = 0
        for keyword_1 in paper_info_1['keywords']:
            for keyword_2 in paper_info_2['keywords']:
                if keyword_1.lower().strip() == keyword_2.lower().strip():
                    same_keyword += 1
                    break
            if same_keyword >= 1:
                break
        if same_keyword >= 1:
            result.append(1) # has same keyword
        else:
            result.append(0)
        # result.append(2 * same_keyword / (len(paper_info_1['keywords']) + len(paper_info_2['keywords']))) # number of same keyword
    except:
        result = [0, 0, 0, 0, 0]

    return result


if __name__ == "__main__":
    with open('data/track2/train/train_unass_data.json', 'r') as r:
        train_unass_data = json.load(r)
    with open('data/track2/train/train_existing_data.json', 'r') as r:
        train_existing_data = json.load(r)
    with open('data/track2/train/train_pub.json', 'r') as r:
        train_pub = json.load(r)

    existing_data_hash_by_name = {}
    for person_id in train_existing_data:
        real_name = train_existing_data[person_id]['name']
        replaced_real_name = real_name.strip().replace('_', ' ')
        if replaced_real_name not in existing_data_hash_by_name:
            existing_data_hash_by_name[replaced_real_name] = {}
        existing_data_hash_by_name[replaced_real_name][person_id] = train_existing_data[person_id]['papers']

    train_x = []
    train_y = []
    for unass_data in train_unass_data:
        unass_paper_id = unass_data[0][:8]
        author_rank = int(unass_data[0][9:])
        unass_author_id = unass_data[1]
        unass_paper_info = train_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank-1]['name']
        for same_name_author_id in existing_data_hash_by_name[the_author_name.lower().strip()]:
            one_person_sim_list = []
            for paper_id in existing_data_hash_by_name[the_author_name.lower().strip()][same_name_author_id]:
                one_person_sim_list.append(compare_two_paper(unass_paper_info, train_pub[paper_id], author_rank))
            train_x.append(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
            if unass_author_id == same_name_author_id:
                train_y.append(1)
            else:
                train_y.append(0)

with open('data/track2/train/train_x.pkl', 'wb') as wb:
    pickle.dump(np.array(train_x), wb)
with open('data/track2/train/train_y.pkl', 'wb') as wb:
    pickle.dump(np.array(train_y), wb)
