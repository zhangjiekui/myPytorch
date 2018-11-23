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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)




model_names= ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]
model_name = "squeezenet"
# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_dir = "hymenoptera_data"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 64

num_workers=4
# Number of epochs to train for
num_epochs = 2
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_val_model(model:nn.Module, dataloaders:dict, criterion:nn.CrossEntropyLoss, optimizer:optim.SGD, num_epochs=num_epochs ,is_inception=False):
    '''
    :param model:
    :param dataloaders: dict ,包括了train和val两个dataloader
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :param is_inception:
    :return:
    '''
    print("************* train_and_valid begined!")
    since= time.time()
    val_acc_history= []
    best_acc= 0.0
    best_model_wts= copy.deepcopy(model.state_dict())

    print('---Epoch train_and_valid begined')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch ,num_epochs-1))
        print('-'*10 )

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  #Set model to training mode
            else:
                model.eval()   #Set model to evaluate mode

            running_loss=0.0
            running_corrects=0

            #Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)

                # zero the parameter
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'): # torch.set_grad_enabled(True) if (phase=='train')
                    if is_inception and phase=='train':
                        outputs, aux_outputs=model(inputs)
                        loss1=criterion(outputs,labels)
                        loss2=criterion(aux_outputs,labels)
                        loss=loss1+0.4*loss2
                    else:
                        outputs=model(inputs)
                        loss=criterion(outputs,labels)
                    _, preds=torch.max(outputs, 1)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                # todo *inputs.size(0)?  因为：The losses are averaged across observations for each minibatch
                # 详见 class CrossEntropyLoss(_WeightedLoss)的定义说明
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)

            epoch_loss=running_loss/len(dataloaders[phase].dataset)
            epoch_acc =running_corrects.double()/len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:4f}'.format(phase,epoch_loss,epoch_acc))
            # deep copy the model
            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
            if phase=='val':
                val_acc_history.append(epoch_acc)
        print('---Epoch {}/{} finished!'.format(epoch ,num_epochs-1))
    time_elapsed=time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    print("************* train_and_valid finished!")
    return model, val_acc_history

def set_parameter_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
# model_name= ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these, variables is model specific.
    model_ft=None
    input_size=0
    if model_name=='resnet':
        model_ft=models.resnet18(pretrained=use_pretrained)
        set_parameter_grad(model_ft,feature_extract)
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=nn.Linear(num_ftrs, num_classes)
        input_size=224

    elif model_name=='alexnet':
        model_ft=models.alexnet(pretrained=use_pretrained)
        set_parameter_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier[6].in_features
        model_ft.classifier[6]=nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name=='vgg':
        model_ft=models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier[6].in_features
        model_ft.classifier[6]=nn.Linear(num_ftrs,num_classes)
        input_size=224

    elif model_name=='squeezenet':
        model_ft=models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_grad(model_ft,feature_extract)
        model_ft.classifier[1]=nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes=num_classes
        input_size=224

    elif model_name=='densenet':
        model_ft=models.densenet121(pretrained=use_pretrained)
        set_parameter_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier.in_features
        model_ft.classifier=nn.Linear(num_ftrs,num_classes)
        input_size=224

    elif model_name=='inception':
        model_ft=models.inception_v3(pretrained=use_pretrained)
        set_parameter_grad(model_ft,feature_extract)
        num_ftrs=model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc=nn.Linear(num_ftrs,num_classes)

        num_ftrs=model_ft.fc.in_features
        model_ft.fc=nn.Linear(num_ftrs,num_classes)
        input_size=299

    else:
        print('Invalid model name, exiting......')
        exit()

    return model_ft, input_size


def print_modelArc():
    # global model_ft, input_size
    for _model_name in model_names:
        model_ft, input_size = initialize_model(_model_name, num_classes, feature_extract, use_pretrained=False)
        print("input_size=", input_size)
        print("_model_name: ", model_ft)


# model_names= ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]
# print_modelArc()

# model_name = "squeezenet"
model_ft, input_size= initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
# Create training and validation dataloaders
dataloaders_dict={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=num_workers) for x in ['train','val']}

model_ft=model_ft.to(device)
params_to_update=model_ft.parameters()

print(model_name)
print("Params to learn:")
if feature_extract:
    params_to_update=[]
    for name,param in model_ft.named_parameters():
        if param.requires_grad==True:
            params_to_update.append(param)
            print('\t', name)

else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad==True:
            print('\t',name)

# Observe that all parameters are being optimized
optimizer_ft=optim.SGD(params_to_update,lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_val_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


# Initialize the non-pretrained version of the model used for this run
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False,use_pretrained=False)
scratch_model_pretrained,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)

scratch_model = scratch_model.to(device)
scratch_model_pretrained=scratch_model_pretrained.to(device)

scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_optimizer = optim.SGD(scratch_model_pretrained.parameters(), lr=0.001, momentum=0.9)

scratch_criterion = nn.CrossEntropyLoss()
scratch_criterion_pretrained = nn.CrossEntropyLoss()

_,scratch_hist = train_val_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
_,scratch_hist_pretrained = train_val_model(scratch_model_pretrained, dataloaders_dict, scratch_criterion_pretrained, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []
phist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]
phist = [h.cpu().numpy() for h in scratch_hist_pretrained]

title=model_name + " Validation Accuracy vs. Number of Training Epochs:"
plt.title(title)
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained: "+str(max(ohist)))
plt.plot(range(1,num_epochs+1),shist,label="Scratch: "+str(max(shist)))
plt.plot(range(1,num_epochs+1),phist,label="scratch_pretrained :"+str(max(phist)))
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

# 0.928105 0.915033
# 0.457516
# 0.915033 0.928105



