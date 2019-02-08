# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:27:57 2018

@author: yanyongyu
"""
import numpy as np
import cv2 as cv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

#处理图片反色
def pre_pic(pic_path,isApp=False,isGray=True):
    #实时使用
    if isApp:
        img = Image.fromarray(cv.cvtColor(pic_path,cv.COLOR_BGR2RGB))
    #打开图片
    else:
        img = Image.open(pic_path)
    #检测图片大小
    if img.size != (11,18):
        raise Exception("Image has the wrong size!Try resize it into (11,18)")
    #是否灰度处理
    if isGray:
        im_arr = np.array(ImageOps.invert(img.convert('L')))
        nm_arr = im_arr.reshape([1,198])
    else:
        im_arr = np.array(ImageOps.invert(img))
        nm_arr = im_arr.reshape([1,198*3])
    '''
    threshold = 65
    for i in range(18):
        for j in range(11):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    '''
    '''
    plt.imshow(im_arr)
    plt.show()
    '''
    '''
    plt.imshow(im_arr,cmap='gray')
    plt.show()
    '''
    #转换为32位
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    return img_ready
    
if __name__ == '__main__':
    img_ready = pre_pic(r'C:\Users\yanyo\Desktop\typeeasy\correct_model\pictures\Annes_best_friend1.0.0.jpg',isGray=False,shape=[1,198*3])
