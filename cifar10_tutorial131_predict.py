# -*- coding: utf-8 -*-
# Author: Zhangjiekui
# Date: 2018-11-5 11:37

'''
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples-download
https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

from __future__ import print_function
from pytorch_tutorial.cifar10_tutorial131 import Net as Net

import torch
import torchvision
from torchvision import transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=200)
Path_dictstate='netState_dict.pth'
Path_model='net.pth'
Path_dictstate_withoutModule='netState_dict_withoutModule.pth'
'''
When saving the model using nn.DataParallel, 
which stores the model in module, and then I was trying to load it without DataParallel.
So, either I need to add a nn.DataParallel temporarily in my network for loading purposes,
 or I can load the weights file, create a new ordered dict without the module prefix, and load it back.
'''
model_dictstate=Net()
model_dictstate_withoutModule=Net()
# original saved file with DataParallel
state_dict=torch.load(Path_dictstate)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dictstate.load_state_dict(new_state_dict)

'''
When saving the model using nn.DataParallel, 
which stores the model in module, and then I was trying to load it without DataParallel.
So, either I need to add a nn.DataParallel temporarily in my network for loading purposes,
or I can load the weights file, create a new ordered dict without the module prefix, and load it back.

------------------------
the best way is:
using
torch.save(model.module.state_dict(), path_to_file)
instead of 
torch.save(model.state_dict(), path_to_file)
that way you don’t get the “module.” string to begin with…

as shown in the next
'''
model_dictstate_withoutModule.load_state_dict(torch.load(Path_dictstate_withoutModule))
model_dictstate.eval()
model_dictstate.to(device)

model_entire=torch.load(Path_model)
model_entire.eval()
model_entire.to(device)

print("model loaded on :", model_dictstate)

num_workers=4
bs=512


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# trainset=torchvision.datasets.CIFAR10(root="./data",train=True, transform=transform,download=True)
# # train_data :(50000, 32, 32, 3)   train_labels: 50000
# trainloader=torch.utils.data.DataLoader(trainset,batch_size=bs,shuffle=True,num_workers=num_workers)

testset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
# test_data :(10000, 32, 32, 3)   test_labels: 10000
testloader=torch.utils.data.DataLoader(testset,batch_size=bs,shuffle=False,num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_cls=len(classes)

total1=0
correct1=0
total2=0
correct2=0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels =images.to(device), labels.to(device)
        outputs1 = model_dictstate(images)
        outputs2 = model_entire(images)
        _, predicted1= torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        total1 += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()
        total2 += labels.size(0)
        correct2 += (predicted1 == labels).sum().item()

print('model_dictstate Accuracy of the network on the test images: %d %%' % (
    100 * correct1 / total1))
print('model_entire Accuracy of the network on the test images: %d %%' % (
    100 * correct2 / total2))

# class_correct = list(0. for i in range(num_cls))
# class_correct2 = list(0. for i in range(num_cls))
# class_total = list(0. for i in range(num_cls))
# class_total2 = list(0. for i in range(num_cls))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = net(images)
#         max, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         # 实际运行时使用下面的for循环会快很多，保留只是为了验证结果的正确性
#         for i in range(labels.size(0)):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#         # 实际运行时使用下面的for循环会快很多，保留只是为了验证结果的正确性
#         for cls in range(num_cls):
#             class_correct2[cls]+=c[labels==cls].sum().item()
#             class_total2[cls] += [labels == cls][0].sum().item()
# print('class_total',class_total)
# print('class_total2',class_total2)
# print('class_correct',class_correct)
# print('class_correct2',class_correct2)

# for i in range(num_cls):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))


