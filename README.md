# myPytorch
自己学习敲的代码

# todo https://github.com/zhangjiekui/Pytorch-Project-Template
# todo https://pytorch.org/tutorials/beginner/saving_loading_models.html  

saved model in cifar10_tutorial131.py, 
load and predict in cifar10_tutorial131_predict.py,  cifar10_tutorial131_predict2.py


torch.save(net, 'net.pth')  -- model_entire=torch.load(Path_model)
torch.save(net.module.state_dict(),  'netState_dict_withoutModule.pth') --         
                                          model_dictstate_withoutModule.load_state_dict(torch.load(Path_dictstate_withoutModule))

torch.save(net.state_dict(), 'netState_dict.pth')  --
state_dict=torch.load(Path_dictstate)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params

model_dictstate.load_state_dict(new_state_dict)
