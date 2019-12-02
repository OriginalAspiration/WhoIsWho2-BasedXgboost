import sys
sys.path.append("/home/rqy/.local/lib/python3.5/site-packages")
import time
import json
import pickle

import numpy as np
import xgboost as xgb

def basic_name(author_name,name):
    tmp_name = name.replace("-"," ").replace("  "," ")
    tmp_author_name = author_name.replace("_"," ")
    if replace_str(tmp_name) == replace_str(tmp_author_name):
        return True
    return False

def name23(author_name,name):
    first_name = None
    second_name = None
    first_author_name = None
    second_author_name = None
    author_name = author_name.split("_")
    author_name_len = len(author_name)
    if author_name_len == 2:
        first_author_name = author_name[1]
        second_author_name = author_name[0]
    if author_name_len == 3:
        first_author_name = author_name[2]
        second_author_name = author_name[0]+author_name[1]
    name = name.replace("-","")
    name = name.split(" ")
    name_len = len(name)
    if name_len == 2:
        first_name = name[1]
        second_name = name[0]
    if name_len == 3:
        first_name = name[2]
        second_name = name[0]+name[1]

    if first_name is None or \
       second_name is None or \
       first_author_name is None or \
       second_author_name is None:
        return False

    if second_name == second_author_name and \
        first_name == first_author_name:
            return True
    if second_name == first_author_name and \
        first_name == second_author_name:
            return True
    return False

def name_fuck(author_name, name):#处理缩写的情况
    first_name = None
    second_name = None
    first_author_name = None
    second_author_name = None
    t = author_name
    t1 = name
    author_name = author_name.split("_")
    author_name_len = len(author_name)
    name = name.replace("  "," ")
    if len(name) == 0:
        return False
    try:
        if name[0] == ' ':
            name= name[1:]
        if author_name_len == 2:
            first_author_name = author_name[1]
            second_author_name = author_name[0][0]
            if author_name_len == 3:
                first_author_name = author_name[2]
                second_author_name = author_name[0][0]+author_name[1][0]
        name = name.replace("-","")
        name = name.split(" ")
        name_len = len(name)
        if name_len == 2:
            first_name = name[1]
            second_name = name[0][0]
        if name_len == 3:
            first_name = name[2]
            second_name = name[0][0]+name[1][0]
    except Exception as e:
        print(t)
        print(t1)
        print(name)
        print(author_name)
        print(name[0])
        exit()

    if first_name is None or \
       second_name is None or \
       first_author_name is None or \
       second_author_name is None:
        return False

    if second_name == second_author_name and \
        first_name == first_author_name:
            return True
    if second_name == first_author_name and \
        first_name == second_author_name:
            return True
    return False

def is_author_name_compress(author_name):
    author_name = author_name.split("_")
    author_name_len = len(author_name)
    if author_name_len == 2 and \
         (len(author_name[0]) == 1 or \
         len(author_name[1]) == 1):
         return True

    if author_name_len == 3 :
        if len(author_name[0]) == 1 and \
        len(author_name[1]) == 1:
            return True
    if len(author_name[1]) == 1 and \
        len(author_name[2]) == 1:
            return True
    return False

def compare_name_old(author_name, name):
    old_name = name
    author_name = author_name.lower()
    name = name.lower().replace("."," ").replace("  "," ")

    #最基础的一种情况
    if basic_name(author_name, name):
        return True
    if name23(author_name, name):
        return True
    if '.' in old_name and name_fuck(author_name, name):
        return True

    return False

zhuanhuanbiao = {
    "L Deng" : ["li_deng"],
    "Wang Ying" : ["haiying_wang"],
    "Li Wen-Gang" : [""],
    "Li Chun" : ["c_c_lin"], #???
    "GUAN Ning-zhang" : [""],
    "ZHAO Shi-hao" : [""],
    "Zongcheng Zhan" : [""],
    "LI Wen-gang" : [""],
    "Zhang Shan-Shan" : [""],
    "LI Chun" : [""],
    "Peng Chong-mei" : [""],
    "zhang rang" : [""],
    "Pan Xue-Qiang" : [""],
    "jianfang" : ["jianfang_wang"],
    "XIN Rong-ya" : [""],
    "Hu Guang" : [""],
    "WANG Zhi-min" : [""],
    "hui" : ["hui_zhang"],
    "qu zhang" : [""],
    "ZHAO Shu-hua" : [""],
    "FENG Jiong-xin" : [""],
    "ZHANG Xiao-yan" : [""],
    "Dr. Hui Xiong" : ["hui_xiong"],
    "Xie D" : ["dan_xie"],
    "ZHANG Yong-xin" : [""],
    "Zhang Xian" : [""],
    "osamu" : ["osamu_watanabe"],
    "bo" : [""],
    "Chen Liang-Mian" : [""],
    "Liu, Ling" : ["ling_liu"],
    "WANG Gui-xian" : [""],
    "XIAO Wang-chuan" : [""],
    "hang" : ["hang_li"],
    "MENG Meng" : ["meng_wang"],  #????
    "K Zhou" : [""],
    "ZHAO Yue" : [""],
    "WU Yi-peng" : [""],
    " " : ["hong_he", "xia_li"],
    "WANG Chuan-li" : [""],
    "WANG Zhen-zhong" : [""],
    "fei" : ["fei_wei"],
    "lin" : ["lin_he"],
    "CUI Guo-xing" : [""],
    "CHEN Liang-mian" : [""],
    "Wang Zhen-Zhong" : [""],
    "FENG Wei-hong" : [""],
    "jianping" : ["jianping_wu", "jianping_ding", "jianping_fan"],
    "jinghong" : ["jinghong_li"],
    "Zhang Wei" : [""],
    "jun" : ["jun_yang"],
    "QIAN Yu" : [""],
    "Michael Zhang" : ["qiwei_zhang"],
    "michael zhang" : ["qiwei_zhang"],
    "Xian Zhang" : [""],
    "Wan Kai-Yang" : [""],
}


def compare_name(author_name, name):
    if name in zhuanhuanbiao and author_name in zhuanhuanbiao[name]:
        return True

    old_name = name
    author_name = author_name.lower()
    name = name.lower().replace("."," ").replace("  "," ")

    #最基础的一种情况
    if basic_name(author_name, name):
        return True
    if name23(author_name, name):
        return True
    if '.' in old_name and name_fuck(author_name, name):
        return True
    if is_author_name_compress(author_name):
        #print('author_name', author_name)
        if name_fuck(author_name, name):
            return True
    return False

def replace_str(input):
    input = input.strip().replace('_', '').replace('-', '').replace(' ', '').replace('.', '').replace(chr(160), '').lower()
    return input.replace('jie yang', 'jieyang').replace('yangjie', 'jieyang').replace('liubing', 'bingliu').replace('0008', '').replace('0002', '').replace('0001', '')


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
