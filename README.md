# myPytorch
自己学习敲的代码


# todo https://pytorch.org/tutorials/beginner/saving_loading_models.html 
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
