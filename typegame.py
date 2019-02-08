# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:19:32 2018

@author: yanyongyu
"""

import win32gui as gui
from PIL import ImageGrab
from matplotlib import pyplot as plt
import numpy as np
import image_processing
import pre_pic
import win32com.client

#
class TypeGame():
    def __init__(self,title=None,isSave=True,isGray=True):
        self.title = title
        self.isSave = isSave
        self.isGray = isGray

    def gui_control(self):
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        gui.SetForegroundWindow(self.hwnd)
        left, top, right, bottom = gui.GetWindowRect(self.hwnd)
        return left, top, right, bottom
    
    def start(self):
        class_name = 'QWidget'
        title_name = '金山打字通 2016'

        self.hwnd = gui.FindWindow(class_name,title_name)
        #print(self.hwnd)
        if self.hwnd:
            #gui.CloseWindow(self.hwnd)
            left, top, right, bottom = self.gui_control()
            if (left<0 or top<0 or right<0 or bottom<0):
                raise Exception("Please don't hide the window!")
            
            left += 45
            top += 155
            right -= 50
            bottom -= 130
            rect = (left, top, right, bottom)
            
            img = ImageGrab.grab().crop(rect)
            '''
            plt.imshow(img)
            plt.show()
            '''
            
            if self.isSave:
                for j in range(5):
                    for i in range(66):
                        img_ready = img.crop((10.4*i,63.8*j,10.4*(i+1),63.8*j+18))
                        plt.imshow(img_ready)
                        plt.xticks([]), plt.yticks([])
                        plt.show()
                        img_ready.save(PICTURE_PATH.format(self.title, j, i))
                        
                        char = input("Please enter the char: ")
                        with open(LABEL_PATH,'a') as f:
                            f.write("./pictures/{}.{}.{}.jpg {}\n".format(self.title, j, i, ord(char)))
                        
            elif self.isGray:
                for j in range(5):
                    for i in range(66):
                        img_croped = img.crop((10.4*i,63.8*j,10.4*(i+1),63.8*j+18))
                        '''
                        plt.imshow(img_croped)
                        plt.xticks([]), plt.yticks([])
                        plt.show()
                        '''
                        img_processed = image_processing.processing(img_croped,False)
                        img_ready = pre_pic.pre_pic(img_processed,isApp=True)
                        if j==0 and i==0:
                            imgs = img_ready
                        else:
                            imgs = np.r_[imgs,img_ready]
                #print(imgs)
                yield self.hwnd,imgs
            else:
                for j in range(5):
                    for i in range(66):
                        img_croped = img.crop((10.4*i,63.8*j,10.4*(i+1),63.8*j+18))
                        '''
                        plt.imshow(img_croped)
                        plt.xticks([]), plt.yticks([])
                        plt.show()
                        '''
                        img_processed = image_processing.processing(img_croped,False)
                        img_ready = pre_pic.pre_pic(img_processed,isApp=True,isGray=False)
                        if j==0 and i==0:
                            imgs = img_ready
                        else:
                            imgs = np.r_[imgs,img_ready]
                #print(imgs)
                yield self.hwnd,imgs
            
        else:
            raise Exception("金山打字通2016 窗口捕获失败!")
            
if __name__ == '__main__':
    '''
    MODEL_PATH = './type_model/'
    LABEL_PATH = MODEL_PATH + 'labels.txt'
    PICTURE_PATH = MODEL_PATH + 'pictures/{}.{}.{}.jpg'
    tg = TypeGame('Annes_best_friend1')
    for _ in tg.start():
        pass
    '''
    '''
    MODEL_PATH = './correct_model/'
    LABEL_PATH = MODEL_PATH + 'labels.txt'
    PICTURE_PATH = MODEL_PATH + 'pictures/{}.{}.{}.jpg'
    tg = TypeGame('Annes_best_friend8')
    for _ in tg.start():
        pass
    '''
    '''
    tg = TypeGame(isSave=False)
    for each in tg.start():
        plt.imshow(each)
        plt.xticks([]), plt.yticks([])
        plt.show()
    '''
    '''
    tg = TypeGame(isSave=False,isOneHot=False)
    function = tg.start()
    for _,each in function:
        imgs = each
        print(imgs)
    '''
