echo "---------- split_data.py ----------"
date
python split_data.py
echo "---------- format_process.py ----------"
date
python format_process.py
echo "---------- train_model.py ----------"
date
python train_model.py
echo "---------- paper2paper_xgb.py ----------"
date
python paper2paper_xgb.py
echo "---------- get_train_data.py ----------"
date
python get_train_data.py
echo "---------- train_xgb.py ----------"
date
python train_xgb.py
python train_nn.py
echo "---------- script.py ----------"
date
python script.py