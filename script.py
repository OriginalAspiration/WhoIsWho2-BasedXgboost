import time
import json
import pickle

import numpy as np
import xgboost as xgb
from tqdm import tqdm


def compare_two_paper(paper_info_1, paper_info_2, author_rank):
    try:
        result = []
        the_author_name = paper_info_1['authors'][author_rank]['name']

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
                if author_2['org'] == paper_info_1['authors'][author_rank]['org']:
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
                if replace_str(keyword_1) == replace_str(keyword_2):
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


def replace_str(input):
    input = input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').lower()
    return input.replace('yangjie', 'jieyang').replace('liubing', 'bingliu').replace('0008', '').replace('0002', '').replace('\xa0', '')


if __name__ == "__main__":
    with open('data/track2/cna_data/cna_valid_unass_competition.json', 'r') as r:
        cna_valid_unass_competition = json.load(r)
    with open('data/track2/cna_data/cna_valid_pub.json', 'r') as r:
        cna_valid_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile_pub.json', 'r') as r:
        whole_author_profile_pub = json.load(r)
    with open('data/track2/cna_data/whole_author_profile.json', 'r') as r:
        whole_author_profile = json.load(r)

    whole_data_hash_by_name = {}
    for person_id in whole_author_profile:
        real_name = whole_author_profile[person_id]['name']
        replaced_real_name = replace_str(real_name)
        if replaced_real_name not in whole_data_hash_by_name:
            whole_data_hash_by_name[replaced_real_name] = {}
        whole_data_hash_by_name[replaced_real_name][person_id] = whole_author_profile[person_id]['papers']

    result_dict = {}
    error_times = 0
    bst = xgb.Booster(model_file='xgb_1.model')
    for index, unass_data in tqdm(enumerate(cna_valid_unass_competition)):
        unass_paper_id = unass_data[:8]
        author_rank = int(unass_data[9:])
        unass_paper_info = cna_valid_pub[unass_paper_id]
        the_author_name = unass_paper_info['authors'][author_rank]['name']

        cna_x = []
        id_list = []
        try:
            for same_name_author_id in whole_data_hash_by_name[replace_str(the_author_name)]:
                one_person_sim_list = []
                for paper_id in whole_data_hash_by_name[replace_str(the_author_name)][same_name_author_id]:
                    one_person_sim_list.append(compare_two_paper(unass_paper_info, whole_author_profile_pub[paper_id], author_rank))
                cna_x.append(np.sum(one_person_sim_list, axis=0) / len(one_person_sim_list))
                id_list.append(same_name_author_id)
            dtest=xgb.DMatrix(np.array(cna_x))
            ypred=bst.predict(dtest)
            predicted_author_id = id_list[np.argsort(ypred)[-1].item()]
            if predicted_author_id not in result_dict:
                result_dict[predicted_author_id] = []
            result_dict[predicted_author_id].append(unass_paper_id)
        except Exception as e:
            error_times += 1
            print('error_times: ', error_times)
            print(e)
            result_dict['laYWugfp'].append(unass_paper_id)
        # if index % 100 == 0:
        #     print('--- index: ' + str(index) + ' / ' + str(len(cna_valid_unass_competition)))

    with open('result.json', 'w') as w:
        w.write(json.dumps(result_dict))

