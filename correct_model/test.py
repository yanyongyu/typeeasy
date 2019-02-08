# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:48:44 2018

@author: yanyongyu
"""
import tensorflow as tf
import time
import forward
import backward
import generateds
import numpy as np
import matplotlib.pyplot as plt

#设置刷新时间
TEST_INTERVAL_SECS = 5
#设置test样本数
TEST_NUM = 2079

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            TEST_NUM,
            forward.IMAGE_SIZE_1,
            forward.IMAGE_SIZE_2,
            forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        #定义准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        generator = generateds.Get_tfrecord(TEST_NUM)
        img_batch, label_batch = generator.get_next()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        while True:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1]\
                                  .split('-')[-1]

                    xs, ys = sess.run([img_batch, label_batch])
                    reshaped_xs = np.reshape(xs,(
                        TEST_NUM,
                        forward.IMAGE_SIZE_1,
                        forward.IMAGE_SIZE_2,
                        forward.NUM_CHANNELS))
                    '''
                    plt.imshow(np.reshape(xs,(forward.IMAGE_SIZE_1,forward.IMAGE_SIZE_2)),cmap='gray')
                    plt.show()
                    pre_y, pre_y_ = sess.run([tf.argmax(y, 1), tf.argmax(y_,1)],feed_dict=\
                                              {x: reshaped_xs,y_: ys})
                    print(pre_y, pre_y_)
                    '''
                    accuracy_score = sess.run(accuracy,feed_dict=\
                                              {x: reshaped_xs,y_: ys})
                    print('After %s training steps,test accuracy = %g' % \
                          (global_step, accuracy_score))
                    
                else:
                    print('No checkpoint file found!')
                    return
            time.sleep(TEST_INTERVAL_SECS)
            
if __name__ == '__main__':
    test()
