import cv2
from PIL import Image
import numpy as np
import copy




p = '/dlprojects/chineseocrpytorch0.41/chineseocr/train/data/ocr/test/inv1.png'
# p ='/dlprojects/chineseocrpytorch0.41/chineseocr/train/data/ocr/test/yyfp.jpg'

import numpy as np
import collections


# 定义字典存放颜色分量上下限
# 例如：{颜色: [min分量, max分量]}
# {'red': [array([160,  43,  46]), array([179, 255, 255])]}

def getColorList():
    print('go in getColorList()')
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list


    # #灰色
    # lower_gray = np.array([0, 0, 46])
    # upper_gray = np.array([180, 43, 220])
    # color_list = []
    # color_list.append(lower_gray)
    # color_list.append(upper_gray)
    # dict['gray']=color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # 红色1
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red1'] = color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # yellow and orange
    lower_yellow = np.array([11, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['ylw&org'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # green,blue,cyan(qing se)
    lower_blue = np.array([35, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['grn&blu&cyn'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict

def ern_dlt(frame,kernel_size=5):
    print('go in ern_dlt()')

    kernel=np.ones((kernel_size,kernel_size),np.uint8)
    ern=cv2.erode(frame,kernel,iterations=2)
    dlt=cv2.dilate(frame,kernel,iterations=1)

    # cv2.imshow('ern',ern)
    cv2.imwrite('dlt.jpg',dlt)
    # cv2.imshow('dlt',dlt)
    # ern_dlt=cv2.dilate(ern,kernel,iterations=2)
    # cv2.imshow('ern_dlt',ern_dlt)
    # opening=cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel)
    # closeing=cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel)
    # cv2.imshow('opening', opening)
    # cv2.imshow('closeing', closeing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 处理图片
def get_color(frame,color_dict,kernel_size=5,iterations=2):
    print('go in get_color()')
    maxsum = -100
    color = None
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # color_dict = getColorList()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for d in color_dict:
        mask_o = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite('color_'+d + '.jpg', mask_o)
        mask = cv2.threshold(mask_o, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(mask, kernel, iterations=iterations)
        cv2.imwrite('color_' + d + '_dlt.jpg', binary)
        img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print('num_contours:',np.size(cnts))
        # print('num_contours[0]:', cnts[0])
        # print("hiera:",hiera)
        if d=='red2':
            # gen the image part in 'red2' color
            white=255
            index=mask==white #white

            white_image = np.zeros(frame.shape, np.uint8)
            white_image[:,:]=(white,white,white)
            white_image[index]=frame[index] #(0,0,255)
            cv2.imwrite('y_white_image.jpg',white_image)

            black_image=np.zeros(frame.shape, np.uint8)
            black_image[index]=frame[index] #(0,0,255)
            cv2.imwrite('y_black_image.jpg',black_image)
            # gen the image part in 'red2' color

            cv2.imwrite('y_img_returnedby_findContours.jpg', img)
            cv2.imwrite('y_dilate_binary.jpg', binary)
            cv2.drawContours(frame, cnts, -1, [222, 222, 222], thickness=-1)
            cv2.imwrite('y_drawContours.jpg', frame)




        sum = 0
        c_max = []
        for c in cnts:
            area=cv2.contourArea(c)
            sum += area
            x, y, w, h = cv2.boundingRect(c)
            if d == 'red2':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite('drawContours_rec.jpg', frame)
                n=15
                newimage = frame[y+n:y+h-n, x+n:x+w-n]  # 先用y确定高，再用x确定宽
                cv2.imwrite(str(y)+'crop_.jpg',newimage)
            print('x, y, w, h',d,x, y, w, h )
            contourArea = cv2.contourArea(c)
            recArea = w*h
            print('contourArea=',cv2.contourArea(c),'recArea=w*h=',w*h,'cont/rec=',contourArea/recArea)

        if sum > maxsum:
            maxsum = sum
            color = d
    return color


if __name__ == '__main__':
    color_dict = getColorList()
    print(color_dict)

    num = len(color_dict)
    print('num=', num)

    for d in color_dict:
        print('key=', d)
        print('value=', color_dict[d][0],color_dict[d][1])

    frame = cv2.imread(p)
    print(get_color(frame,color_dict,kernel_size=3))
    # ern_dlt(frame)


def gen_newImage():
    yuantu_p = '/dlprojects/chineseocrpytorch0.41/chineseocr/train/data/ocr/test/inv1.png'
    contour_p = '/dlprojects/chineseocrpytorch0.41/chineseocr/zjk_testcode/drawContours.jpg'
    yuantu = cv2.imread(yuantu_p)
    contour = cv2.imread(contour_p)
    yuantu2 = copy.deepcopy(yuantu)
    diff = yuantu - contour
    cv2.imwrite('diff.jpg', diff)
    sub = cv2.subtract(contour, yuantu)
    cv2.imwrite('sub.jpg', sub)
    sub_back = cv2.subtract(yuantu, yuantu)
    cv2.imwrite('sub_back1.jpg', sub_back)
    rows, cols, channels = yuantu.shape  # rows，cols
    yuantu2 = yuantu.copy()
    cv2.imwrite('yuantu.jpg', yuantu)
    for i in range(rows):
        for j in range(cols):
            if np.all(contour[i, j] == [222, 222, 222]):  # 0代表黑色的点
                sub_back[i, j] = yuantu[i, j]  # 此处替换颜色，为BGR通道
    cv2.imwrite('y_sub_back2.jpg', sub_back)
    cv2.imwrite('y_yuantu2.jpg', yuantu2)


# gen_newImage() The method code can work, but rubbish


'''
#RGB -Blue, Green, Red
blue=img[:,:,0]
green=img[:,:,1]
red=img[:,:,2]
cv2.imwrite('blue.jpg',blue)
cv2.imwrite('green.jpg',green)
cv2.imwrite('red.jpg',red)

# cv2.imshow('blue',blue)
# cv2.waitKey(0)
# cv2.imshow('green',green)
# cv2.waitKey(0)
# cv2.imshow('red',red)
# cv2.waitKey(0)

b=img.copy()
b[:,:,1]=0
b[:,:,2]=0
g=img.copy()
g[:,:,0]=0
g[:,:,2]=0
r=img.copy()
r[:,:,0]=0
r[:,:,1]=0
cv2.imwrite('blueonly.jpg',b)
cv2.imwrite('greenonly.jpg',g)
cv2.imwrite('redonly.jpg',r)


# cv2.imshow('blue',b)
# cv2.waitKey(0)
# cv2.imshow('green',g)
# cv2.waitKey(0)
# cv2.imshow('red',r)
# cv2.waitKey(0)
'''
