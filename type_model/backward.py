# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:26:26 2018

@author: yanyongyu
"""
import tensorflow as tf
import os
import numpy as np
from . import forward
from . import generateds

#训练次数
STEPS = 5000
#每一轮输入样本数
BATCH_SIZE = 50
NUM_EXAMPLES = 2309
#学习率梯度下降
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
#正则化
REGULARIZER = 0.0001
#滑动平均值
MOVING_AVERAGE_DECAY = 0.99
#模型断点保存
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'typeeasy_model'

#定义反向传播函数
def backward():
    with tf.name_scope('inputs'):
        #输入数据占位
        x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                        forward.IMAGE_SIZE_1,
                                        forward.IMAGE_SIZE_2,
                                        forward.NUM_CHANNELS],name='image_data')
        #标签数据占位
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE],name='image_num')
    #前向传播框架
    y = forward.forward(x, True, REGULARIZER)
    #初始化计步器
    global_step = tf.Variable(0,trainable=False)
    
    with tf.name_scope('loss'):
        #正则化交叉熵损失函数
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cem = tf.reduce_mean(ce)
        loss = cem +tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss',loss)

    #学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,\
                                               global_step,\
                                               NUM_EXAMPLES / BATCH_SIZE,\
                                               LEARNING_RATE_DECAY,\
                                               staircase=True)
    with tf.name_scope('train'):
        #定义训练优化方法
        train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                     .minimize(loss,global_step=global_step)

    #初始化滑动平均值
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    #初始化保存器
    saver = tf.train.Saver()
    
    #初始化数据集读取器
    generator = generateds.Get_tfrecord(BATCH_SIZE)
    img_batch, label_batch = generator.get_next()

    #分配GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #tensorboard
        writer = tf.summary.FileWriter('logs/',sess.graph)
        merged = tf.summary.merge_all()

        #读取模型记录
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #线程协调器开启
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            reshaped_xs = np.reshape(xs,(
                BATCH_SIZE,
                forward.IMAGE_SIZE_1,
                forward.IMAGE_SIZE_2,
                forward.NUM_CHANNELS))
            _, loss_value, step, result = sess.run([train_op, loss, global_step, merged], feed_dict={x: reshaped_xs ,y_: ys})
            if i % 100 == 0:
                print("After %d training steps, loss on training batch is: %g" % (step,loss_value))
                writer.add_summary(result,i)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)

        #线程协调器关闭
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    backward()
