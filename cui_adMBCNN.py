#最好效果
import os
import time
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold
from torch.nn import init
from torch_geometric import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset,Data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
    precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout,Flatten, ReLU, \
    MaxPool3d, Conv3d,Conv2d,MaxPool2d
import torch.nn.functional as F
import scipy.io as sio

##########################################################
"""
Settings for training 
"""
classes = 2  # Num. of classes
if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')
def Split_Sets_Fold(total_fold, data):

    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True)
    if len(data) == 0:
            for i in range(total_fold):
                train_index.append([])
                test_index.append([])
    else:
        for train_i, test_i in kf.split(data):
            train_index.append(train_i)
            test_index.append(test_i)
    return train_index, test_index

drop_out = 0.3

class cui_adMBCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlpE = nn.Sequential(
            Conv3d(1, 32, kernel_size=(3, 3, 3),padding=1),
            ReLU(),
            MaxPool3d(kernel_size=(2,2, 2), stride=2),
            Conv3d(32, 64, kernel_size=(3,3,3),padding=1),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            Flatten(),
            Dropout(drop_out),
            Linear(64 * 7 *7, 50),
        )
        self.mlpV0 = nn.Sequential(
            Conv3d(1, 32, kernel_size=(3, 3, 3),padding=2),
            ReLU(),
            MaxPool3d(kernel_size=(2,2, 2), stride=2), #3 4 3
            Conv3d(32, 64, kernel_size=(3,3,3),padding=2), #5 6 5
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2), #2 3 2
            Flatten(),
            Dropout(drop_out),
            Linear(64 * 2 * 3 *2, 50),
        )
        self.mlpV1 = nn.Sequential(
            Conv3d(1, 32, kernel_size=(3, 3, 3), padding=2),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # 3 4 3
            Conv3d(32, 64, kernel_size=(3, 3, 3), padding=2),  # 5 6 5
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # 2 3 2
            Flatten(),
            Dropout(drop_out),
            Linear(64 * 2 * 3 * 2, 50),
        )
        self.mlpV2 = nn.Sequential(
            Conv3d(1, 32, kernel_size=(3, 3, 3), padding=2),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # 3 4 3
            Conv3d(32, 64, kernel_size=(3, 3, 3), padding=2),  # 5 6 5
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # 2 3 2
            Flatten(),
            Dropout(drop_out),
            Linear(64 * 2 * 3 * 2, 50),
        )

        self.drop0 = Dropout(drop_out)
        self.linend = nn.Sequential(
            Linear(200,2),
            )
    def forward(self, data):
        batch_size = len(data)
        adj = data.edge.reshape(batch_size,1,4,30,30)#404 4 30 30
        adj0 = self.mlpE(adj)
        x0 = self.mlpV0(data.de.reshape(batch_size, 1, 4, 7, 5))
        x1 = self.mlpV1(data.psd.reshape(batch_size, 1, 4, 7, 5))
        x2 = self.mlpV2(data.fe.reshape(batch_size, 1, 4, 7, 5))
        xx = torch.cat((x0,x1,x2,adj0), axis=1)
        xx = self.drop0(xx)
        xx = self.linend(xx)
        yy = F.softmax(xx,1)
        return xx,yy


def train(model, train_loader, crit, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        label =  torch.from_numpy(np.array(data.y).ravel())
        label = label.to(device)  # , dtype=torch.long) #, dtype=torch.int64)
        output, _ = model(data)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)  # train_datast


def evaluate(model, loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            #label = data.y.view(-1, classes)
            label =  torch.from_numpy(np.array(data.y).ravel())
            #label =  torch.Tensor(data.y)
            data = data.to(device)
            _, pred = model(data)
            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.hstack(labels)
    try:
        AUC = roc_auc_score(labels, np.argmax(predictions, axis=-1), average='macro')
    except:
        AUC = 0
    precision, recall, _, support = precision_recall_fscore_support(labels,
                                                                    np.argmax(predictions, axis=-1), zero_division=0)
    f1 = f1_score(labels, np.argmax(predictions, axis=-1), average='macro')
    acc = accuracy_score(labels, np.argmax(predictions, axis=-1))
    return AUC, acc, f1, recall, precision,support


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
time_start = time.time()

Network00 = cui_adMBCNN
result_data = []
all_last_acc = []
all_last_AUC = []
epochs = 250
sumlist = []
epoch_data = []
max_list = []
setup_seed(1)

def main_E(data,edge,labels):
    dataset = []
    for i in range(len(labels)):
        dataset.append(Data(de=data[0][i],psd=data[1][i],fe=data[2][i],edge = edge[i], y=labels[i]))

    for fold in range(5):
        train_dataset = []
        test_dataset = []

        train_index, test_index = Split_Sets_Fold(5, dataset)
        for i in train_index[fold]:
            train_dataset.append(dataset[i])

        for i in test_index[fold]:
            test_dataset.append(dataset[i])
        train_loader = DataLoader(train_dataset, batch_size=100, drop_last=False, shuffle=True,generator=torch.Generator().manual_seed(1))
        test_loader = DataLoader(test_dataset, batch_size=2022, drop_last=False, shuffle=False)
        model = Network00().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        crit = torch.nn.CrossEntropyLoss()  #

        themax = 0.0
        themax_recall = 0.0
        themax_precision = 0.0
        # 开始训练
        for epoch in range(epochs):
            t0 = time.time()
            loss = train(model, train_loader, crit, optimizer)
            train_AUC, train_acc, train_f1, train_recall, train_precision,support = evaluate(model, train_loader)
            t1 = time.time()
            print(
                'f{:01d} ,EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f},Time: {:.2f}'.format(
                    fold, epoch + 1, loss, train_AUC, train_acc,  (t1 - t0)))

            if epcoch == 250:
                test_AUC, test_acc, test_f1, test_recall, test_precision, test_support = evaluate(model, test_loader)
                themax = test_acc
                themax_recall = test_recall[1]
                themax_precision = test_precision[1]
            #更新学习率
            #scheduler.step()
        print(fold, '--the max acc is', themax)
        max_list.append(themax)
    print("文件" + str(version), Network00.__name__)
    print(max_list)
    print(sum(max_list)/5)
time_start = time.time()
# #读取数据
m_edge = sio.loadmat(r"cui/滤波后数据/totalspearman.mat")
edge = m_edge['totalspearman']
edge= torch.from_numpy(edge).to(device)
edge = edge.permute(3,2,0,1)

m_de = sio.loadmat('cui/节点熵值/DE.mat')
m_fe = sio.loadmat('cui/节点熵值/FE.mat')
m_psd = sio.loadmat('cui/节点熵值/PSD.mat')
labels = m_de['label']
de = m_de['DE']
fe = m_fe['FE']
psd = m_psd['PSD']
de = de.transpose(2, 1, 0)
fe = fe.transpose(2, 1, 0)
de = torch.from_numpy(de).to(device) #30,4,2022
fe = torch.from_numpy(fe).to(device)
psd = torch.from_numpy(psd).to(device) #2022,4,30
new = torch.zeros(3, 2022, 4, 7, 5).to(device)
for num, feature in zip([0, 1, 2], [de, psd, fe]):
    for i in range(2022):
        new[num, i, :, 0, 1] = feature[i, :, 0]
        new[num, i, :, 0, 3] = feature[i, :, 1]
        new[num, i, :, 1, :] = feature[i, :, 2:7]
        new[num, i, :, 2, :] = feature[i, :, 7:12]
        new[num, i, :, 3, :] = feature[i, :, 12:17]
        new[num, i, :, 4, :] = feature[i, :, 17:22]
        new[num, i, :, 5, :] = feature[i, :, 22:27]
        new[num, i, :, 6, 1:4] = feature[i, :, 27:30]
main_E(new,edge,labels)
print("共计耗时：", (time.time() - time_start) / 60, "分钟")