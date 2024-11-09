import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

net = nn.Sequential(nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1),nn.BatchNorm2d(96),nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3,stride=2),
                        nn.Conv2d(96,256,kernel_size=5,padding=2),nn.BatchNorm2d(256),nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3,stride=2),
                        nn.Conv2d(256,384,kernel_size=3,padding=1),nn.BatchNorm2d(384),nn.ReLU(),
                        nn.Conv2d(384,384,kernel_size=3,padding=1),nn.BatchNorm2d(384),nn.ReLU(),
                        nn.Conv2d(384,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3,stride=2),
                        nn.Flatten(),
                        nn.Linear(6400,4096),nn.BatchNorm1d(4096),nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096,2048),nn.BatchNorm1d(2048),nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(2048,10))

def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight.data)

net = net.to(device='cuda')

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), 
])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=256,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=256,shuffle=False,num_workers=0)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
for epoch in range(20):
    running_loss = 0.0
    number = 0
    for X,y in trainloader:
        optimizer.zero_grad()
        l = loss(net(X.cuda()),y.cuda()) 
        l.backward()
        optimizer.step()
        running_loss += l.item()
        number += 1
    print(epoch,': ',running_loss / number)
print('Finish')

correct = 0
num = 0
with torch.no_grad():
    for X,y in testloader:
        num += y.size(0)
        _ , predicted = torch.max(net(X.cuda()),1)
        correct += (predicted == y.cuda()).sum().item()
result = 100 * (correct / num)

print(result)