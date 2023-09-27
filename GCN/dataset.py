# coding=utf-8
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

odata = pd.read_csv('./dataset/7.11D.csv')
graph = pd.read_csv('./dataset/D_graph.csv')


X = odata[['S_chn_ppp_20', 'S_road_dis', 'S_busdis', 'S_subdis',
       'S_jun_dis', 'S_shp_Large volume', 'S_shp_country', 'S_shp_crowd',
       'S_shp_empty', 'S_shp_road', 'S_shp_squire', 'S_fun_交通用地', 'S_fun_公共服务',
       'S_fun_商业用地', 'S_fun_基础设施用地', 'S_fun_居住用地', 'S_fun_工业用地', 'S_fun_空地',
       'S_fun_绿地', 'S_shpstd', 'S_funstd', 'S_junction_num', 'S_marketjoin',
       'S_busjoin', 'S_length']].to_numpy()
print(X.shape)
y = odata[['count']].to_numpy()
# y = y.reshape(-1)
print(y.shape)

x = torch.tensor(X,dtype=torch.float)
y = torch.tensor(y,dtype=torch.float)


gra = graph[['InputID' ,'TargetID']].to_numpy().T
weight = graph[['Distance']].to_numpy().T
weight = weight.reshape(-1)

gra = torch.tensor(gra,dtype=torch.long)
weight = torch.tensor(weight,dtype=torch.float)

train_mask = []
test_mask = []

for i in range(y.shape[0]):
    if i<=543:
        a = True
        b = False
    else:
        a = False
        b = True
    train_mask.append(a)
    test_mask.append(b)
train_mask = torch.tensor(train_mask,dtype=torch.bool)
test_mask = torch.tensor(test_mask,dtype=torch.bool)


data = Data(x=x,y=y,edge_index=gra,edge_attr=weight,train_mask = train_mask,test_mask = test_mask).to(device)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(25, 16)
        self.conv2 = GCNConv(16, 1)
        # self.fc = nn.Sequential(
        #     nn.Linear(1,677),
        #     nn.ELU(),
        #     nn.Linear(677,1)
        # )

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        # x = self.fc(x)
        # x = torch.flatten(x)

        return x

#
# # 5.3) 训练 & 测试.

model = Net().to(device)

print(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    print(data.y.size())
    print(out.size())
    loss = F.mse_loss(out, data.y)

    loss.backward()
    optimizer.step()



# model.eval()
# _, pred = model(data)
print(model(data))

print(data.y)
# correct = float(pred.eq(data.y).sum().item())
# acc = correct / data.sum().item()
# print('Accuracy: {:.4f}'.format(acc))