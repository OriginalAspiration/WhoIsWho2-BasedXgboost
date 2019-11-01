# Whoiswho 比赛

## 赛道2
赛道2需要将数据放置在data/track2/train和data/track2/cna_data目录下


* **split_data.py**:用于将完整的训练数据切分成“现有数据库”和“新增论文”两部分，模拟赛道2的场景

* **get_train_data.py**: 用于提取特征，为训练xgboost准备数据

* **train_xgb.py**: 用于训练xgboost模型，模型保存为xgb_n.model

* **count_name.py**: 用于统计新增论文与现有数据库中姓名不匹配的情况，用于辅助整理name字段

* **script.py**: 根据cna_data生成最终结果result.json

* **name_of_whole_data.json**: 包括数据库中姓名数据，用于辅助整理name字段
