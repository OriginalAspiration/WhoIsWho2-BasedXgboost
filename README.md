# Whoiswho 比赛

The run method:

sh run.sh

## 赛道2
赛道2需要将数据放置在data/track2/train和data/track2/cna_data目录下


* **split_data.py**:用于将完整的训练数据切分成“现有数据库”和“新增论文”两部分，模拟赛道2的场景

* **format_process.py**: 用于预处理数据，完成分词，词干化以及预训练tf_idf

* **train_model.py**: 用于训练nltk及gensim的模型，并保存模型信息

* **paper2paper_xgb.py**: 用于训练paper to paper的xgboost模型，并保存模型信息

* **store_keywords_map.py**: 用于获取关键词之间的关联信息（关键词1与关键词2是否相关）

* **get_train_data.py**: 用于提取特征，为训练xgboost准备数据，compare_two_paper函数列出了全部特征

* **train_xgb.py**: 用于训练xgboost模型，模型保存为xgb_n.model

* **count_name.py**: 用于统计新增论文与现有数据库中姓名不匹配的情况，以及辅助整理name字段

* **script.py**: 根据cna_data生成最终结果result.json

* **post_process.py**: 调整准确率和召回率比例（使二者尽可能相近），获得最佳F1分数



输出文件格式（部分）

| 输出文件 | 生成文件 | 格式 |
| - | - | - |
| train_existing_data.json | split_data.py | 论文id-该论文在作者下的编号 作者 |
| train_unass_data | split_data.py | 作者id,下面是name和论文id列表|
| train_pub_alter.json | format_process.py | |
| train_abstract_tf_idf.txt | |
| train_tf_idf.txt | |
| train_word_dict.txt | |
| train_abstract_word_dict.txt | |


备注：如果需要复现最佳结果，需要切换到train_p2p分支，用该版本计算的特征训练p2p_xgb模型，并将模型训练的结果data/track2/train/train_pub_p2p_result_title.res和data/track2/test/test_pub_p2p_result_title.res考本到本分支的对应位置，删除run.sh中python paper2paper_xgb.py这一行，并将script.py中INIT_P2P_XGB置为False。
使用上述方法可以将F1得分提升约0.005，我们猜测是因为train_p2p分支使用的是更加简单的特征，这些特征对于p2p_xgb模型效果更好。如果觉得麻烦可以忽略这一备注，对最终结果和理解本模型不会有显著的影响。如果对于模型运行方法有任何疑问或有任何建议，可以发送邮件至baiwt@bupt.edu.cn
