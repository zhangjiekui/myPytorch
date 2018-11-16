# -*- coding: utf-8 -*-
# Author: Zhangjiekui
# Date: 2018-11-16 15:39
# torch.set_printoptions(linewidth=200)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net=Net()
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)
# net.to(device)
from __future__ import print_function