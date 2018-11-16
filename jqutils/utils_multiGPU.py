# -*- coding: utf-8 -*-
# Author: Zhangjiekui
# Date: 2018-11-15 23:52
# torch.set_printoptions(linewidth=200)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net=Net()
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)
# net.to(device)
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

err_savemode_msg="错误：目前只支持['state_dict','entire']两种模型存储模式！"
err_none_model_msg="错误：必须提供模型类示例！"

def using_multiGPUs(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_GPU=torch.cuda.device_count()
    if num_GPU > 1:
      print("Using", num_GPU, "GPUs!")
      model = nn.DataParallel(model)
    model.to(device)
    return model, num_GPU

def save_model(model,saveMode='state_dict', num_GPU=2, savePathName='netState_dict_withoutModule.pth'):
    '''
    :param model: 需要存储的模型
    :param saveMode: 'state_dict', 'entire'
    :param num_GPU: 训练模型时使用的GPU数量
    :param savePathName: 模型存储后的路径名称
    :return:
    '''
    if saveMode=='state_dict':
        if num_GPU>0:
            torch.save(model.module.state_dict(), savePathName)
        else:
            torch.save(model.state_dict(),savePathName)
    elif saveMode=='entire':
        torch.save(model, savePathName)
    else:
        print(err_savemode_msg)
        raise NotImplementedError()
    return saveMode,savePathName

def load_model(saveMode='state_dict',savePathName='netState_dict_withoutModule.pth',model=None, device=None):
    '''
    :param saveMode: 'state_dict', 'entire'
    :param savePathName: 模型存储后的路径名称
    :param model: 需要装载模型的存储路径文件
    :param device: 模型装载到GPU或是CPU，由实例化后的device指定

    :return:
    '''
    if saveMode=='state_dict':
        if model==None:
            print(err_none_model_msg)
            raise NotImplementedError()
        else:
            model.load_state_dict(torch.load(savePathName))
    elif saveMode=='entire':
        model=torch.load(savePathName)
    else:
        print(err_savemode_msg)
        raise NotImplementedError()
    if device!=None:
        model.to(device)
    return model
