# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:19:15 2018

@author: yanyongyu
"""
import tensorflow as tf

IMAGE_SIZE_1 = 18
IMAGE_SIZE_2 = 11
NUM_CHANNELS = 3
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 2

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def forward(x,train,regularizer):
    with tf.name_scope('conv1'):
        with tf.name_scope('weights'):
            conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM],regularizer)
            tf.summary.histogram('conv1/weights',conv1_w)
        with tf.name_scope('biases'):
            conv1_b = get_bias([CONV1_KERNEL_NUM])
            tf.summary.histogram('conv1/biases',conv1_b)
        with tf.name_scope('conv2d'):
            conv1 = conv2d(x, conv1_w)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
        with tf.name_scope('pool1'):
            pool1 = max_pool_2x2(relu1)

    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE,CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer)
            tf.summary.histogram('conv2/weights',conv2_w)
        with tf.name_scope('biases'):
            conv2_b = get_bias([CONV2_KERNEL_NUM])
            tf.summary.histogram('conv2/biases',conv2_b)
        with tf.name_scope('conv2d'):
            conv2 = conv2d(pool1,conv2_w)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
        with tf.name_scope('pool2'):
            pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.name_scope('fc1'):
        with tf.name_scope('weights'):
            fc1_w = get_weight([nodes,FC_SIZE], regularizer)
        with tf.name_scope('biases'):
            fc1_b = get_bias([FC_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.3)

    with tf.name_scope('fc2'):
        with tf.name_scope('weights'):
            fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
        with tf.name_scope('biases'):
            fc2_b = get_bias([OUTPUT_NODE])
        with tf.name_scope('matmul'):
            y = tf.matmul(fc1, fc2_w) + fc2_b
        return y
