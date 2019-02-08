# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:51:40 2018

@author: yanyongyu
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os

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
        img = img.convert('L')
        img = ImageOps.invert(img)
        img_raw = img.tobytes()
        labels = [0] * 79
        if 48 <= img_label <= 57:#0~9对应数字0~9
            labels[img_label-48] = 1
        elif 65 <= img_label <= 90:#10~35对应A~Z
            labels[img_label-55] = 1
        elif 97 <= img_label <= 122:#36~61对应a~z
            labels[img_label-61] = 1
        elif img_label == 32:#62对应空格
            labels[62] = 1
        elif img_label == 33:#63对应!
            labels[63] = 1
        elif img_label == 34:#64对应"
            labels[64] = 1
        elif img_label == 35:#65对应#
            labels[65] = 1
        elif img_label == 36:#66对应$
            labels[66] = 1
        elif img_label == 37:#67对应%
            labels[67] = 1
        elif img_label == 38:#68对应&
            labels[68] = 1
        elif img_label == 39:#69对应'
            labels[69] = 1
        elif img_label == 40:#70对应(
            labels[70] = 1
        elif img_label == 41:#71对应)
            labels[71] = 1
        elif img_label == 44:#72对应,
            labels[72] = 1
        elif img_label == 45:#73对应-
            labels[73] = 1
        elif img_label == 46:#74对应.
            labels[74] = 1
        elif img_label == 47:#75对应/
            labels[75] = 1
        elif img_label == 58:#76对应:
            labels[76] = 1
        elif img_label == 59:#77对应;
            labels[77] = 1
        elif img_label == 63:#78对应?
            labels[78] = 1
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
                                               'label':tf.FixedLenFeature([79],tf.int64)
                                               })
            img = tf.decode_raw(features['img_raw'], tf.uint8)
            img.set_shape([198])
            img =  tf.cast(img, tf.float32) * (1./255)
            label = tf.cast(features['label'],tf.float32)
            return img, label
        
        dataset = tf.data.TFRecordDataset(tfRecord_path)
        dataset = dataset.map(_parser).repeat().shuffle(buffer_size=1000).batch(self.batch_size)
        
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
    img = tf.reshape(img,[18,11])
    #分配GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        img, label = sess.run([img,label])
        plt.imshow(img,cmap='gray')
        plt.show()
        print(label)
