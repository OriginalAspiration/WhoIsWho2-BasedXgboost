import json
import pickle

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split


HIDDEN_NUM = 50
LR = 0.01
BATCH_SIZE = 50000
eps = np.finfo(np.float32).eps.item()


class Net1(nn.Module):

    def __init__(self, input_num):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(input_num, HIDDEN_NUM)
        self.fc2 = nn.Linear(HIDDEN_NUM, 2 * HIDDEN_NUM)
        self.fc3 = nn.Linear(2 * HIDDEN_NUM, HIDDEN_NUM)
        self.fc4 = nn.Linear(HIDDEN_NUM, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.out_act(x)
        return x


if __name__ == "__main__":
    with open('data/track2/train/train_x.pkl', 'rb') as rb:
        train_x = pickle.load(rb)
    with open('data/track2/train/train_y.pkl', 'rb') as rb:
        train_y = pickle.load(rb)

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,test_size = 0.2,random_state = 1)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)

    train_torch_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
    )

    net = Net1(train_x.shape[-1])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss_func = torch.nn.BCELoss()

    for epoch in range(20):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x
            batch_y = batch_y
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            print('epoch: ' + str(epoch) + ' - step: ' + str(step) + ' - RMSE: ' + str(loss.pow(0.5).item()))
        scheduler.step()

    ypred = net(test_x).squeeze().detach().numpy()
    y_pred = (ypred >= 0.5)*1
    from sklearn import metrics
    print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
    print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
    print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
    print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
    print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
    print(metrics.confusion_matrix(test_y,y_pred))

    torch.save(net, 'nn_1.model')

