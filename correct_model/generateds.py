# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:36:18 2018

@author: yanyongyu
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

image_train_path = './pictures/'
label_train_path = './labels.txt'
tfRecord_train = './data/typeeasy.tfrecords'
data_path = './data/'
resize_height = 18
resize_width = 11

def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img_path = value[0]
        img_label = int(value[1])
        img = Image.open(img_path)
        img = ImageOps.invert(img)
        img_raw = img.tobytes()
        labels = [0] * 2
        if img_label == 48:#0对应灰色
            labels[0] = 1
        elif img_label == 49:#1对应红色
            labels[1] = 1
        else:
            print(img_path,img_label)

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print('the number of picture: ',num_pic)
    writer.close()
    print('write tfrecord successfully')

def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.mkdir(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)

class Get_tfrecord():
    def __init__(self, batch_size, isTrain=True):
        self.batch_size = batch_size
        if isTrain:
            self.tfRecord_path = tfRecord_train
        else:
            pass
            #self.tfRecord_path = tfRecord_test
        self.read_tfRecord(tfRecord_train, self.batch_size)
            
    def read_tfRecord(self, tfRecord_path, batch_size):
        #旧版读取方法
        '''
        filename_queue = tf.train.string_input_producer([tfRecord_path])
        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_examples,
                                       features={
                                           'img_raw':tf.FixedLenFeature([],tf.string),
                                           'label':tf.FixedLenFeature([79],tf.int64)})
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img.set_shape([198])
        img =  tf.cast(img, tf.float32) * (1./255)
        label = tf.cast(features['label'],tf.float32)
        '''
        #新版dataset读取方法
        def _parser(record):
            features = tf.parse_single_example(record,
                                           features={
                                               'img_raw':tf.FixedLenFeature([],tf.string),
                                               'label':tf.FixedLenFeature([2],tf.int64)
                                               })
            img = tf.decode_raw(features['img_raw'], tf.uint8)
            img.set_shape([594])
            img =  tf.cast(img, tf.float32) * (1./255)
            label = tf.cast(features['label'],tf.float32)
            return img, label
        
        dataset = tf.data.TFRecordDataset(tfRecord_path)
        dataset = dataset.map(_parser).repeat().shuffle(buffer_size=1000).batch(batch_size)
        
        self.iterator = dataset.make_one_shot_iterator()
    
    def get_next(self):
        img, label = self.iterator.get_next()
        return img, label

def main():
    generate_tfRecord()

if __name__ == '__main__':
    #main()
    generator = Get_tfrecord(1)
    img, label = generator.get_next()
    img = tf.reshape(img,[18,11,3])
    #分配GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        img, label = sess.run([img,label])
        plt.imshow(img,cmap='gray')
        plt.show()
        print(label)
