# Whoiswho 比赛

Now I have add a model to compare with two papers, the F1 score on borad is 77.7.

The run method:

python split_data.py

python format_process.py

python paper2paper_xgb.py.py

python get_train_data.py

python train_xgb.py

python script.py

## 赛道2
赛道2需要将数据放置在data/track2/train和data/track2/cna_data目录下


* **split_data.py**:用于将完整的训练数据切分成“现有数据库”和“新增论文”两部分，模拟赛道2的场景

* **format_process.py**: 用于预处理数据，完成分词，词干化以及预训练tf_idf

* **get_train_data.py**: 用于提取特征，为训练xgboost准备数据

* **train_xgb.py**: 用于训练xgboost模型，模型保存为xgb_n.model

* **count_name.py**: 用于统计新增论文与现有数据库中姓名不匹配的情况，用于辅助整理name字段

* **script.py**: 根据cna_data生成最终结果result.json

* **name_of_whole_data.json**: 包括数据库中姓名数据，用于辅助整理name字段


输出文件格式

| 输出文件 | 生成文件 | 格式 |
| - | - | - |
| train_existing_data.json | split_data.py | 论文id-该论文在作者下的编号 作者 |
| train_unass_data | split_data.py | 作者id,下面是name和论文id列表|
| train_pub_alter.json | format_process.py | |
| train_abstract_tf_idf.txt | |
| train_tf_idf.txt | |
| train_word_dict.txt | |
| train_abstract_word_dict.txt | |
