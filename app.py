# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:40:28 2018

@author: yanyongyu
"""
import tensorflow as tf
import numpy as np

import win32gui as gui
import win32api as api
import win32con as con
import pythoncom

from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk, ImageOps
import threading
import time
import matplotlib.pyplot as plt

import type_model.forward as t_forward
import type_model.backward as t_backward
import correct_model.forward as c_forward
import correct_model.backward as c_backward
import typegame

#字母识别模块
T_MODEL_SAVE_PATH = './type_model/model'
#加强训练模块
C_MODEL_SAVE_PATH = './correct_model/model'
#pywin32键盘字符映射字典
VK_CODE = {'backspace':0x08,'tab':0x09,'clear':0x0C,'enter':0x0D,'shift':0x10,'ctrl':0x11,'alt':0x12,
           'pause':0x13,'caps_lock':0x14,'esc':0x1B,'spacebar':0x20,'page_up':0x21,'page_down':0x22,
           'end':0x23,'home':0x24,'left_arrow':0x25,'up_arrow':0x26,'right_arrow':0x27,'down_arrow':0x28,
           'select':0x29,'print':0x2A,'execute':0x2B,'print_screen':0x2C,'ins':0x2D,'del':0x2E,'help':0x2F,
           '0':0x30,'1':0x31,'2':0x32,'3':0x33,'4':0x34,'5':0x35,'6':0x36,'7':0x37,'8':0x38,
           '9':0x39,'a':0x41,'b':0x42,'c':0x43,'d':0x44,'e':0x45,'f':0x46,'g':0x47,'h':0x48,
           'i':0x49,'j':0x4A,'k':0x4B,'l':0x4C,'m':0x4D,'n':0x4E,'o':0x4F,'p':0x50,'q':0x51,
           'r':0x52,'s':0x53,'t':0x54,'u':0x55,'v':0x56,'w':0x57,'x':0x58,'y':0x59,'z':0x5A,
           'numpad_0':0x60,'numpad_1':0x61,'numpad_2':0x62,'numpad_3':0x63,'numpad_4':0x64,
           'numpad_5':0x65,'numpad_6':0x66,'numpad_7':0x67,'numpad_8':0x68,'numpad_9':0x69,
           'multiply_key':0x6A,'add_key':0x6B,'separator_key':0x6C,'subtract_key':0x6D,
           'decimal_key':0x6E,'divide_key':0x6F,
           'F1':0x70,'F2':0x71,'F3':0x72,'F4':0x73,'F5':0x74,'F6':0x75,'F7':0x76,'F8':0x77,
           'F9':0x78,'F10':0x79,'F11':0x7A,'F12':0x7B,'F13':0x7C,'F14':0x7D,'F15':0x7E,
           'F16':0x7F,'F17':0x80,'F18':0x81,'F19':0x82,'F20':0x83,'F21':0x84,'F22':0x85,
           'F23':0x86,'F24':0x87,'num_lock':0x90,'scroll_lock':0x91,'left_shift':0xA0,
           'right_shift ':0xA1,'left_control':0xA2,'right_control':0xA3,'left_menu':0xA4,'right_menu':0xA5,
           'browser_back':0xA6,'browser_forward':0xA7,'browser_refresh':0xA8,'browser_stop':0xA9,
           'browser_search':0xAA,'browser_favorites':0xAB,'browser_start_and_home':0xAC,'volume_mute':0xAD,
           'volume_Down':0xAE,'volume_up':0xAF,'next_track':0xB0,'previous_track':0xB1,'stop_media':0xB2,
           'play/pause_media':0xB3,'start_mail':0xB4,'select_media':0xB5,'start_application_1':0xB6,
           'start_application_2':0xB7,'attn_key':0xF6,'crsel_key':0xF7,'exsel_key':0xF8,'play_key':0xFA,
           'zoom_key':0xFB,'clear_key':0xFE,
           '+':0xBB,',':0xBC,'-':0xBD,'.':0xBE,'/':0xBF,'`':0xC0,';':0xBA,
           '[':0xDB,'\\':0xDC,']':0xDD,"'":0xDE,'`':0xC0}

#加强训练线程
class CorrectThread(threading.Thread):
    def __init__(self,mainthread,textVar,master):
        threading.Thread.__init__(self)
        #主线程
        self.mainthread = mainthread
        self.textVar = textVar
        self.master = master
        self.isRunning = True
        self.remain_correct = None
        self.correct_label = None

    def run(self):
        self.top = Toplevel(self.master)
        self.top.title('Correct Model')
        self.top.bind('<Key-Return>',lambda x: self.correct())
        self.top.focus_set()
        frame = Frame(self.top)
        frame.pack(fill=BOTH)
        Label(frame,text='是否开始纠正错误？',style='F1.TLabel').grid(row=0,column=0,columnspan=2,sticky=W+E,padx=10,pady=5)
        Button(frame,text='是',style='F1.TButton',command=self.correct).grid(row=1,column=0,sticky=W+E,padx=10,pady=5)
        Button(frame,text='否',style='F1.TButton',command=self.cancel).grid(row=1,column=1,sticky=W+E,padx=10,pady=5)

    #关闭窗口
    def cancel(self,event=None):
        if not event:
            self.top.destroy()
        else:
            event.widget.master.master.destroy()

    #确认加强训练
    def correct(self):
        #self.top.destroy()
        self.top.children['!frame'].children['!label']['text'] = "是否开始反向传播？"
        self.top.children['!frame'].children['!button']['command'] = ""
        game = typegame.TypeGame(isSave=False,isGray=False)
        tg = game.start()

        #显示错误图片
        for _,self.img in tg:
            preValue = self.restore_model(self.img)
            self.show(preValue)

    #预测是否错误
    def restore_model(self,testPicArr):
        with tf.Graph().as_default() as tg:
            x = tf.placeholder(tf.float32, [330,
                                            c_forward.IMAGE_SIZE_1,
                                            c_forward.IMAGE_SIZE_2,
                                            c_forward.NUM_CHANNELS])
            y = c_forward.forward(x, False, None)
            preValue = tf.argmax(y, 1)

            variable_averages = tf.train.ExponentialMovingAverage(c_backward.MOVING_AVERAGE_DECAY)
            variable_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variable_to_restore)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                reshaped_xs = np.reshape(testPicArr,(
                    330,
                    c_forward.IMAGE_SIZE_1,
                    c_forward.IMAGE_SIZE_2,
                    c_forward.NUM_CHANNELS))
                ckpt = tf.train.get_checkpoint_state(C_MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    preValue = sess.run(preValue, feed_dict={x:reshaped_xs})
                    return preValue
                else:
                    print('No checkpoint file found!')
                    return -1

    #显示图片
    def show(self,preValue):
        self.top_list = {}
        self.mainthread.wrong_list = []
        for i in range(len(preValue)):
            if preValue[i] == 1:
                print(i)
                img_array = self.img[i].reshape([18,11,3])*255.0
                img1 = Image.fromarray(img_array.astype('uint8')).convert('RGB')
                img1 = ImageOps.invert(img1)
                img_gray = img1.convert('L')
                '''
                plt.imshow(img1)
                plt.show()
                '''
                width, height = img1.size
                resized_img = img1.resize((width*5, height*5),Image.ANTIALIAS)
                tk_img = ImageTk.PhotoImage(resized_img)
                self.mainthread.wrong_list.append(tk_img)
                self.top1 = Toplevel(self.top)
                self.top1.title('Correct Model')
                frame = Frame(self.top1)
                frame.pack(fill=BOTH)
                label = Label(frame,image=tk_img,style='F1.TLabel',anchor='center')
                label.grid(row=0,column=0,columnspan=2,sticky=W+E,padx=10,pady=5)
                label['image'] = tk_img
                self.v1 = StringVar()
                entry = Entry(frame,textvariable=self.v1)
                entry.grid(row=1,column=0,columnspan=2,sticky=W+E,padx=10,pady=5)
                entry.focus_set()
                button1 = Button(frame,text='确定',style='F1.TButton')
                self.top1.bind('<Key-Return>',lambda event : self.backward_model(event))
                button1.bind('<ButtonRelease-1>',lambda event : self.backward_model(event))
                button1.grid(row=2,column=0,sticky=W+E,padx=10,pady=5)
                button2 = Button(frame,text='取消',style='F1.TButton')
                button2.grid(row=2,column=1,sticky=W+E,padx=10,pady=5)
                button2.bind('<ButtonRelease-1>',lambda event : self.cancel(event))
                self.top_list[self.top1] = np.array(img_gray).reshape([1,198]).astype(np.float32)

    #将图片及标签加入数组
    def backward_model(self,event):
        label = event.widget.master.children['!entry'].get()
        if label:
            try:
                label = ord(label)
                labels = np.zeros([1,79],dtype=np.float32)
                if 48 <= label <= 57:#0~9对应数字0~9
                    labels[0][label-48] = 1
                elif 65 <= label <= 90:#10~35对应A~Z
                    labels[0][label-55] = 1
                elif 97 <= label <= 122:#36~61对应a~z
                    labels[0][label-61] = 1
                elif label == 32:#62对应空格
                    labels[0][62] = 1
                elif label == 33:#63对应!
                    labels[0][63] = 1
                elif label == 34:#64对应"
                    labels[0][64] = 1
                elif label == 35:#65对应#
                    labels[0][65] = 1
                elif label == 36:#66对应$
                    labels[0][66] = 1
                elif label == 37:#67对应%
                    labels[0][67] = 1
                elif label == 38:#68对应&
                    labels[0][68] = 1
                elif label == 39:#69对应'
                    labels[0][69] = 1
                elif label == 40:#70对应(
                    labels[0][70] = 1
                elif label == 41:#71对应)
                    labels[0][71] = 1
                elif label == 44:#72对应,
                    labels[0][72] = 1
                elif label == 45:#73对应-
                    labels[0][73] = 1
                elif label == 46:#74对应.
                    labels[0][74] = 1
                elif label == 47:#75对应/
                    labels[0][75] = 1
                elif label == 58:#76对应:
                    labels[0][76] = 1
                elif label == 59:#77对应;
                    labels[0][77] = 1
                elif label == 63:#78对应?
                    labels[0][78] = 1
                else:
                    print(label,"is not visible!")
                if np.all(self.remain_correct!=None):
                    self.remain_correct = np.r_[self.remain_correct,np.multiply(self.top_list[event.widget.master.master],1.0/255.0)]
                else:
                    self.remain_correct = self.top_list[event.widget.master.master]
                if np.all(self.correct_label!=None):
                    self.correct_label = np.r_[self.correct_label,labels]
                else:
                    self.correct_label = labels
                event.widget.master.master.destroy()
            except Exception as ex:
                print('try again!')
                print(ex)
        else:
            print('please enter the character!')

#扫描线程
class ScanThread(threading.Thread):
    def __init__(self,mainthread,textVar,pause,cancel,master):
        threading.Thread.__init__(self)
        #主线程
        self.mainthread = mainthread
        self.textVar = textVar
        self.master = master
        self.pause = pause
        self.isRunning = True
        self.cancel = cancel

    def run(self):
        #pywin32多线程初始化
        pythoncom.CoInitialize()
        game  = typegame.TypeGame(isSave=False)
        tg = game.start()
        
        for hwnd,each in tg:
            img_ready = each
            preValue = self.restore_model(img_ready)
            text = self.preValue_to_text(preValue)                
            self.textVar.set('正在输入。。。\n'+text)
            #获取窗口位置并单击输入区
            left, top, _, _ = game.gui_control()
            cursor_rect = (left+50,top+190)
            api.SetCursorPos(cursor_rect)
            self.mouse_left_click()
            #开始键盘输入
            if CORRECT:
                for i in text[:-1]:
                    if 65 <= ord(i) <= 90:#A~Z
                        self.key_event(i.lower(),True)
                    elif i == ' ':#空格
                        self.key_event('spacebar')
                    elif i == '!':#!
                        self.key_event('1',True)
                    elif i == '"':#"
                        self.key_event("'",True)
                    elif i == ':':#:
                        self.key_event(';',True)
                    elif i == '?':#?
                        self.key_event('/',True)
                    else:
                        self.key_event(i)
                thread = CorrectThread(self.mainthread,self.textVar,self.master)
                thread.start()
                '''
                i = text[-1]
                if 65 <= ord(i) <= 90:#A~Z
                    self.key_event(i.lower(),True)
                elif i == ' ':#空格
                    self.key_event('spacebar')
                elif i == '!':#!
                    self.key_event('1',True)
                elif i == '"':#"
                    self.key_event("'",True)
                elif i == ':':#:
                    self.key_event(';',True)
                elif i == '?':#?
                    self.key_event('/',True)
                else:
                    self.key_event(i)
                '''
            else:
                for i in text:
                    if 65 <= ord(i) <= 90:#A~Z
                        self.key_event(i.lower(),True)
                    elif i == ' ':#空格
                        self.key_event('spacebar')
                    elif i == '!':#!
                        self.key_event('1',True)
                    elif i == '"':#"
                        self.key_event("'",True)
                    elif i == ':':#:
                        self.key_event(';',True)
                    elif i == '?':#?
                        self.key_event('/',True)
                    else:
                        self.key_event(i)
        self.cancel()

    def stop(self):
        self.isRunning = False

    #预测字母
    def restore_model(self,testPicArr):
        with tf.Graph().as_default() as tg:
            x = tf.placeholder(tf.float32, [330,
                                            t_forward.IMAGE_SIZE_1,
                                            t_forward.IMAGE_SIZE_2,
                                            t_forward.NUM_CHANNELS])
            y = t_forward.forward(x, False, None)
            preValue = tf.argmax(y, 1)

            variable_averages = tf.train.ExponentialMovingAverage(t_backward.MOVING_AVERAGE_DECAY)
            variable_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variable_to_restore)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                reshaped_xs = np.reshape(testPicArr,(
                    330,
                    t_forward.IMAGE_SIZE_1,
                    t_forward.IMAGE_SIZE_2,
                    t_forward.NUM_CHANNELS))
                ckpt = tf.train.get_checkpoint_state(T_MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    preValue = sess.run(preValue, feed_dict={x:reshaped_xs})
                    return preValue
                else:
                    print('No checkpoint file found!')
                    return -1

    #转换标签为字符串
    def preValue_to_text(self,array):
        text = ''
        for preValue in array:
            if 0 <= preValue <= 9:#0~9对应数字0~9
                text += chr(preValue+48)
            elif 10 <= preValue <= 35:#10~35对应A~Z
                text += chr(preValue+55)
            elif 36 <= preValue <= 61:#36~61对应a~z
                text += chr(preValue+61)
            elif preValue == 62:#62对应空格
                text += chr(32)
            elif preValue == 63:#63对应!
                text += chr(33)
            elif preValue == 64:#64对应"
                text += chr(34)
            elif preValue == 65:#65对应#
                text += chr(35)
            elif preValue == 66:#66对应$
                text += chr(36)
            elif preValue == 67:#67对应%
                text += chr(37)
            elif preValue == 68:#68对应&
                text += chr(38)
            elif preValue == 69:#69对应'
                text += chr(39)
            elif preValue == 70:#70对应(
                text += chr(40)
            elif preValue == 71:#71对应)
                text += chr(41)
            elif preValue == 72:#72对应,
                text += chr(44)
            elif preValue == 73:#73对应-
                text += chr(45)
            elif preValue == 74:#74对应.
                text += chr(46)
            elif preValue == 75:#75对应/
                text += chr(47)
            elif preValue == 76:#76对应:
                text += chr(58)
            elif preValue == 77:#77对应;
                text += chr(59)
            elif preValue == 78:#78对应?
                text += chr(63)
            else:
                raise Exception
        #去除过多空格和末尾空格节省时间
        return text.replace('    ','').strip(' ')

    #键盘输入
    def key_event(self,input_key,ifShift=False):
        if ifShift:
            api.keybd_event(16, 0, 0, 0)
        api.keybd_event(VK_CODE[input_key], 0, 0, 0)
        api.keybd_event(VK_CODE[input_key], 0, con.KEYEVENTF_KEYUP, 0)
        if ifShift:
            api.keybd_event(16, 0, con.KEYEVENTF_KEYUP, 0)

    #鼠标左键单击
    def mouse_left_click(self):
        api.mouse_event(con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        api.mouse_event(con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

#主线程
class Main():
    def __init__(self):
        self.text = '当前：未进行任何识别。。。\n请在开始识别前打开金山打字通并请勿最小化\n请自行切换输入法!'
        self.root  = Tk()
        self.root.resizable(0,0)
        self.root.title('金山打字神  --designed by ShowTime Joker')
        self.root.bind('<Key-Return>',lambda event : self.scan())
        style = Style()
        style.configure('TFrame',background='white')
        style.configure('F1.TLabel',background='white',font=('微软雅黑',15))
        style.configure('F1.TButton',background='white',font=('微软雅黑',15))
        self.v1 = StringVar()
        self.v1.set(self.text)
        frame1 = Frame(self.root)
        frame1.pack()
        Label(frame1,textvariable=self.v1,width=40,wraplength=450,style='F1.TLabel').grid(row=0,column=0,padx=10,pady=5,columnspan=3)
        self.button1 = Button(frame1,text='开始识别',style='F1.TButton',command=self.scan)
        self.button1.grid(row=1,column=0,padx=10,pady=5,sticky=W+N+S)
        self.button2 = Button(frame1,text='暂停',state='disabled',style='F1.TButton',command=self.pause_thread)
        self.button2.grid(row=1,column=1,padx=10,pady=5,sticky=N+S)
        self.button3 = Button(frame1,text='取消',state='disabled',style='F1.TButton',command=self.cancel)
        self.button3.grid(row=1,column=2,padx=10,pady=5,sticky=E+N+S)
        self.root.mainloop()

    #开始扫描
    def scan(self):
        self.button1['state'] = 'disabled'
        self.button2['state'] = '!disabled'
        self.button3['state'] = '!disabled'
        self.v1.set('正在扫描。。。\n请勿随意移动/点击鼠标、键盘')
        self.pause = threading.Event()
        self.pause.set()
        self.t1 = ScanThread(self,self.v1,self.pause,self.cancel,self.root)
        #self.t1.setDaemon(True)
        self.t1.start()

    def pause_thread(self):#暂停无效
        self.pause.clear()
        self.v1.set('已暂停')
        self.button2['text'] = '继续'
        self.button2['command'] = self.resume_thread

    def resume_thread(self):#暂停无效
        self.pause.set()
        self.v1.set('正在扫描。。。\n请勿随意移动/点击鼠标、键盘')
        self.button2['text'] = '暂停'
        self.button2['command'] = self.pause_thread

    def cancel(self):#取消无效
        self.t1.stop()
        self.button1['state'] = '!disabled'
        self.button2['state'] = 'disabled'
        self.button3['state'] = 'disabled'
        self.v1.set(self.text)

if __name__ == '__main__':
    CORRECT = False
    Main()
    '''
    root = Tk()
    thread = CorrectThread(None,root)
    thread.start()
    root.mainloop()
    '''
