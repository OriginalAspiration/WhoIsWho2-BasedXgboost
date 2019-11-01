import json
import random

with open('data/track2/train/train_author.json', 'r') as r:
    train_author = json.load(r)
with open('data/track2/train/train_pub.json', 'r') as r:
    train_pub = json.load(r)

unass_data = []
existing_data = {}

for real_name in train_author:
    for person_id in train_author[real_name]:
        papers_of_person = len(train_author[real_name][person_id])
        if papers_of_person >= 2:
            paper_rank = random.randint(0, papers_of_person - 1)
            the_paper_id = train_author[real_name][person_id][paper_rank]
            authors_of_the_paper = train_pub[the_paper_id]['authors']
            for index, authors_info in enumerate(authors_of_the_paper):
                if authors_info['name'].lower().strip() == real_name.strip().replace('_', ' '):
                    unass_data.append((the_paper_id + '-' + str(index), person_id))
                    existing_data[person_id] = {
                        "name": real_name, "papers": [paper_id for paper_id in train_author[real_name][person_id] if not paper_id == the_paper_id]}

with open('data/track2/train/train_unass_data.json', 'w') as w:
    w.write(json.dumps(unass_data))
with open('data/track2/train/train_existing_data.json', 'w') as w:
    w.write(json.dumps(existing_data))