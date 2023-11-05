import torch
import matplotlib.pyplot as plt
from torch.optim import *
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
lr_list = []
for epoch in range(100):
    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')