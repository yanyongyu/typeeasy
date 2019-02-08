# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:10:48 2018

@author: yanyongyu
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#归一化图片大小为18*11
def processing(pic=None,isSaved=True):
    #处理已保存图片并保存
    if isSaved:
        with open(LABEL_PATH,'r') as f:
            pictures = f.readlines()
            
        for each in pictures:
            each = each.split(' ')
            img_path = each[0]
            img_label = each[1]
            img = cv.imread(img_path,1)
            if img.shape == (18,10,3):
                constant = cv.copyMakeBorder(img,0,0,0,1,cv.BORDER_CONSTANT,value=[255,255,255])
                '''
                plt.imshow(constant)
                plt.title('CONSTANT')
                plt.show()
                '''
                cv.imwrite(img_path,constant)
            elif img.shape == (18,11,3):
                pass
            else:
                print(img.shape)
                #raise Exception("Unexpected picture size")
    #处理实时图片
    else:
        img = cv.cvtColor(np.asarray(pic),cv.COLOR_RGB2BGR)
        if img.shape == (18,10,3):
            constant = cv.copyMakeBorder(img,0,0,0,1,cv.BORDER_CONSTANT,value=[255,255,255])
            '''
            plt.imshow(constant)
            plt.title('CONSTANT')
            plt.show()
            '''
            return constant
        elif img.shape == (18,11,3):
            return img
        else:
            print(img.shape)
            #raise Exception("Unexpected picture size")
        
if __name__ == '__main__':
    LABEL_PATH = './labels.txt'
    processing()
