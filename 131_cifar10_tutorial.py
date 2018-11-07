# -*- coding: utf-8 -*-
# Author: Zhangjiekui
# Date: 2018-11-5 11:37
from __future__ import print_function
import torch
import torchvision
from torchvision import transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=200)
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# https://pytorch.org/tutorials/beginner/saving_loading_models.html  #todo

bs=512
num_workers=2
lr=0.001
mm=0.9
epochs=10

trainset=torchvision.datasets.CIFAR10(root="./data",train=True, transform=transform,download=True)
# train_data :(50000, 32, 32, 3)   train_labels: 50000
trainloader=torch.utils.data.DataLoader(trainset,batch_size=bs,shuffle=True,num_workers=num_workers)

testset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
# test_data :(10000, 32, 32, 3)   test_labels: 10000
testloader=torch.utils.data.DataLoader(testset,batch_size=bs,shuffle=False,num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_cls=len(classes)


def imshow(img,prefix):
    img=(img/2+0.5)
    npimg=img.numpy()
    print(npimg.shape)
    npimg_=np.transpose(npimg,(1,2,0))
    print(npimg_.shape)
    plt.imshow(npimg_)
    plt.savefig(prefix+('npimg_.jpg'))

dataiter=iter(trainloader)
imgs, labels=next(dataiter)
# imgs_,labels__=dataiter.next()
imshow(imgs[0],'single')
print(classes[labels.data[0]])

imshow(torchvision.utils.make_grid(imgs),'train')
print(" ".join("%5s" % classes[labels[j]] for j in range(bs)))
# str = "-"
# seq = ("a", "b", "c") # 字符串序列
# print(str.join( seq ))
# result: a-b-c

plt.show()

# Define a Convolution Neural Network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,(5,5))
        self.maxpool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,(5,5))
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,num_cls)
    def forward(self, x):
        print("\tIn Model: input size", x.size())

        # print("inputs.shape=",x.shape)
        # x=self.conv1(x)
        # print("self.conv1(x).shape=",x.shape)
        # x=F.relu(x)
        # print("relu(conv1(x)).shape=", x.shape)
        # x=self.maxpool(x)
        # print("maxpool(relu(conv1(x))).shape=", x.shape)
        x=self.maxpool(F.relu(self.conv1(x)))  #todo 如果不需要打印shape，可以注释掉上面的全部语句

        # x=self.conv2(x)
        # print("self.conv2(x).shape=",x.shape)
        # x=F.relu(x)
        # print("relu(conv2(x)).shape=", x.shape)
        # x=self.maxpool(x)
        # print("maxpool(relu(conv2(x))).shape=", x.shape)
        x=self.maxpool(F.relu(self.conv2(x)))  #todo 如果不需要打印shape，可以注释掉上面的全部语句
        # print("layer2 shape=",x.shape)
        x=x.view(-1,16*5*5)
        # print("x.view shape=", x.shape)
        x=F.relu(self.fc1(x))
        # print("F.relu1(x.view) shape=", x.shape)
        x=F.relu(self.fc2(x))
        # print("F.relu2(x.view) shape=", x.shape)
        x=self.fc3(x)
        # print("fc3(x.view) shape=", x.shape)
        print("\tIn Model: output size", x.size())
        return x

# 多GPU数据并行处理
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #这行是需要的
print("device:", device)
net=Net()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)
# 多GPU数据并行处理

# Define a Loss Function and optimizer
import torch.optim as optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=mm)

for epoch in range(epochs):
    running_loss=0.0
    print("----------in training enumerate-----------------")

    for i, data in enumerate(trainloader, 0):
        inputs,labels=data # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() # zero the parameter gradients
        # forward + backward + optimize
        outputs=net(inputs)
        print("Outside: input size", inputs.size(),"output_size", outputs.size())
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('epoch=',epoch+1,' i=',i+1,' loss=',loss.item(),' inputs.shape=', inputs.shape,' labels.shape=', labels.shape,' trained imgs=',(i+1)*bs )




        # print statistics
        running_loss += loss.item()
        if i % 20==19: # print every 20 mini-batches
            print('[epoch %d, batch %5d]——loss: %.3f' % (epoch+1,i+1 ,running_loss/20))
            running_loss=0.0
print("----------in training enumerate-----------------")
print('Finished Training')


print('-------------------预测咯--------------------------------------')
dataiter=iter(testloader)
testimgs,testlabels=dataiter.next()
imshow(torchvision.utils.make_grid(testimgs),'test')
print('T: '," ".join("%5s" % classes[testlabels[j]] for j in range(bs)))
testimgs,testlabels=testimgs.to(device),testlabels.to(device)
outputs=net(testimgs)
print("predict outputs.shape=",outputs.shape)
max, predicted = torch.max(outputs, 1)
print('P: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(bs)))

print("outputs",outputs[0:10][:].data)
print("max",max[0:10])
print("predicted",predicted[0:10])
p=[predicted[j] for j in range(10)]
print(p)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels =images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(num_cls))
class_correct2 = list(0. for i in range(num_cls))
class_total = list(0. for i in range(num_cls))
class_total2 = list(0. for i in range(num_cls))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        max, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        # 实际运行时使用下面的for循环会快很多，保留只是为了验证结果的正确性
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        # 实际运行时使用下面的for循环会快很多，保留只是为了验证结果的正确性
        for cls in range(num_cls):
            class_correct2[cls]+=c[labels==cls].sum().item()
            class_total2[cls] += [labels == cls][0].sum().item()
print('class_total',class_total)
print('class_total2',class_total2)
print('class_correct',class_correct)
print('class_correct2',class_correct2)

for i in range(num_cls):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
