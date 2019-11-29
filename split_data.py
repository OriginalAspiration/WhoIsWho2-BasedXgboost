import json
import random
import re

random.seed(0)

with open('data/track2/cna_data/whole_author_reformat_like_train.json', 'r') as r:
    train_author = json.load(r)
with open('data/track2/cna_data/whole_author_profile_pub.json', 'r') as r:
    train_pub = json.load(r)

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
cnt = 0
unass_data = []
existing_data = {}

for real_name in train_author:
    if len(train_author[real_name]) > 2:
        for person_id in train_author[real_name]:
            papers_of_person = len(train_author[real_name][person_id])
            #print('papers_of_person', papers_of_person)
            #sum += min(20, papers_of_person) * papers_of_person

            if papers_of_person >= 2:
                for i in range( min(20, papers_of_person) ):
                    if papers_of_person <= 20:
                        paper_rank = i
                    else:
                        paper_rank = random.randint(0, papers_of_person - 1) 
                    the_paper_id = train_author[real_name][person_id][paper_rank]
                    authors_of_the_paper = train_pub[the_paper_id]['authors']
                    for index, authors_info in enumerate(authors_of_the_paper):
                        if fix_name(authors_info['name']) == fix_name(real_name):
                            cnt += 1
                            unass_data.append((the_paper_id + '-' + str(index), person_id))
                            existing_data[person_id] = {
                               "name": real_name, "papers": [paper_id for paper_id in train_author[real_name][person_id]]}

print('len unass_data', len(unass_data))

# unass_data -> 随机抽一篇
# existing_data -> 其余的加入训练集
with open('data/track2/train/train_unass_data.json', 'w') as w:
    w.write(json.dumps(unass_data))
with open('data/track2/train/train_existing_data.json', 'w') as w:
    w.write(json.dumps(existing_data))