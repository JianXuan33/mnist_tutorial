# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:20:46 2019

@author: liujiaxuan
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim
import time

import numpy as np


#import torch.nn.functional as F

#import sys

lr_init = 0.001

#sys.path.append("..")

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(1,1,1)),  normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

   
net = SimpleNet()


# TODO:define loss function and optimiter
ignored_params = list(map(id, net.fc3.parameters()))#fc3

base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer=optim.SGD([
        {'params': base_params},
         {'params': net.fc3.parameters(), 'lr': lr_init*10}],  lr_init,momentum=0.9,weight_decay=1e-4)#fc3
    
       
criterion = nn.CrossEntropyLoss()  

# train and evaluate
for epoch in range(NUM_EPOCHS):
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    for images, labels in tqdm(train_loader):
        # TODO:forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()
        
        
    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset
    loss_sigma = 0.0
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
    net.eval()
    for images, labels in tqdm(train_loader):
        # forward
        outputs = net(images)
        outputs.detach_()
        # 计算loss
        loss = criterion(outputs, labels)
        loss_sigma += loss.item()
        # 统计
        _, predicted = torch.max(outputs.data, 1)
        # labels = labels.data    # Variable --> tensor
        # 统计混淆矩阵
        for j in range(len(labels)):
            cate_i = labels[j].numpy()
            pre_i = predicted[j].numpy()
            conf_mat[cate_i, pre_i] += 1.0
    print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
print('Finished Training')    
    
#print('Training accuracy: %0.2f%%' % (train_accuracy*100))
#print('Testing accuracy: %0.2f%%' % (test_accuracy*100))    
    