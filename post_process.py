import json
import pickle
import numpy as np
with open('data/track2/cna_data/cna_valid_unass_competition.json', 'r') as r:
    cna_valid_unass_competition = json.load(r)

with open('data/track2/cna_data/whole_author_profile.json', 'r') as r:
    whole_author_profile = json.load(r)

with open('result2.json', 'r') as file:
    result_dict = json.load(file)

with open('score_dicts.pkl', 'rb') as file:
    score_dict = pickle.load(file)

print('len cna_valid_unass_competition', len(cna_valid_unass_competition))

LEN = 0

for x, y in result_dict.items():
    LEN += len(y)

print('LEN', LEN)

cnt = 0
kk = set()
for unass_paper_id, sd in score_dict.items():
    ypred = sd['ypred']
    id_list = sd['id_list']

    y1 = ypred[np.argsort(ypred)[-1].item()]
    predicted_id = id_list[np.argsort(ypred)[-1].item()]

    for pred, id in zip(ypred, id_list):
        if pred >= y1-0.005 and predicted_id != id:
            cnt += 1

            result_dict.setdefault(id, [])
            result_dict[id].append(unass_paper_id)

            #print('unass_paper_id',unass_paper_id, 'id', id, 'predicted_id', predicted_id)
            kk.add(unass_paper_id)
print('cnt', cnt)
print('len kk ', len(kk))

'''
#result_dict
author_id_dict = {}
for unass_paper_id, sd in score_dict.items():
    ypred = sd['ypred']
    id_list = sd['id_list']

    for pred, id in zip(ypred, id_list):
        author_id_dict.setdefault(id, [])
        author_id_dict[id].append( (pred, unass_paper_id) )

#print('len author_id_dict', len(author_id_dict))
#print('len result_dict', len(result_dict))

for id,sd in author_id_dict.items():
    if id not in result_dict:
        sd.sort()
        result_dict[id] = [ sd[0][1] ]

        print('id', id, sd[0][1])'''

with open('result3.json', 'w') as file:
    json.dump(result_dict, file)