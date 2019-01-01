cd /data/fastaiV1/fastaiClassV3/

jupyter notebook --no-browser --port=8889

ssh -N -f -L localhost:8889:localhost:8889 zjk@10.1.56.7

or

jupyter notebook --no-browser --port=8896 --allow-root

ssh -N -f -L localhost:8896:localhost:8896 root@192.168.45.48

192.168.45.48

# some definitions

Parameters are the numbers inside the matrices that we multiply by.

An activation is the result of either a matrix multiply or an activation function.

A layer is the result of a matrix multiply + 【input layer】+ 【output layer】

Activation function : such as relu, softmax



# myPytorch
自己学习敲的代码


# NLP
BERT相关论文、文章和代码资源汇总  http://www.52nlp.cn/bert-paper-%E8%AE%BA%E6%96%87-%E6%96%87%E7%AB%A0-%E4%BB%A3%E7%A0%81%E8%B5%84%E6%BA%90%E6%B1%87%E6%80%BB

NLP系列：原理与实践  https://www.jianshu.com/u/abfe703a00fe 


# todo https://pytorch.org/tutorials/beginner/saving_loading_models.html 
       Intro to Deep Learning with PyTorch - Udacity   https://classroom.udacity.com/courses/ud188/lessons/b4ca7aaa-b346-43b1-ae7d-20d27b2eab65/concepts/501ce6c5-9b80-4536-8754-f7da607fc40d  
       Neural Network Programming - Deep Learning with PyTorch  http://deeplizard.com/learn/video/k4jY9L8H89U
# todo 
       SAVING & LOADING A GENERAL CHECKPOINT FOR INFERENCE AND/OR RESUMING TRAINING

saved model in cifar10_tutorial131.py, 
load and predict in cifar10_tutorial131_predict.py,  cifar10_tutorial131_predict2.py

    ------------------------
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
    '''

torch.save(net, 'net.pth')  -- model_entire=torch.load(Path_model)
torch.save(net.module.state_dict(),  'netState_dict_withoutModule.pth') --         
                                          model_dictstate_withoutModule.load_state_dict(torch.load(Path_dictstate_withoutModule))

torch.save(net.state_dict(), 'netState_dict.pth')  --
state_dict=torch.load(Path_dictstate)

    ------------------------
    '''create new OrderedDict that does not contain `module.`'''
    
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
    
        ------------------------
    '''load params'''

model_dictstate.load_state_dict(new_state_dict)

# todo https://github.com/zhangjiekui/Pytorch-Project-Template
