# -*- coding: utf-8 -*-
# Author: Zhangjiekui
# Date: 2018-11-29 13:46
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

# encoding: utf-8
import cv2
import numpy as np
import roi_merge as roi_
import util_funs as util
from get_rects import *


def main(img):
    region = get_rects(img)
    roi_solve = roi_.Roi_solve(region)
    roi_solve.rm_inside()
    roi_solve.rm_overlop()
    region = roi_solve.merge_roi()
    region = util.sort_region(region)
    region = util.get_targetRoi(region)

    for i in range(2):
        rect2 = region[i]
        w1, w2 = rect2[0], rect2[0] + rect2[2]
        h1, h2 = rect2[1], rect2[1] + rect2[3]
        box = [[w1, h2], [w1, h1], [w2, h1], [w2, h2]]
        cv2.drawContours(img, np.array([box]), 0, (0, 255, 0), 1)
        if i == 0:
            cv2.imwrite('代码' + str(i) + '.jpg', img[h1:h2, w1:w2]) #todo k--> i
        else:
            cv2.imwrite('号码' + str(i) + '.jpg', img[h1:h2, w1:w2])
            # cv2.imwrite('号码' + str(k) + '.jpg', img[h1:h2, w1:w2])
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread("inv1.png")
    main(img)
