#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    :version: V1.1.1.18-12-29
    :author: 刘利平
    :file: capsule_network.py
    :time: 18-12-29
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist')

plt.figure(figsize=(10, 3))

for i in range(5):
    plt.subplot(1, 5, i + 1)

    sample_img = mnist.train.images[i].reshape(28, 28)
    plt.imshow(sample_img, cmap="binary")

plt.axis("off")
plt.show()

print('1111')

# 定义数据x和标签y
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32,  name='img')
y = tf.placeholder(shape=[None], dtype=tf.int64, name='y')

# 卷积层参数
with tf.name_scope('conv1_layer'):
    conv1 = tf.layers.conv2d(X, filters=256,
                             kernel_size=9,
                             strides=1,
                             padding='valid',
                             activation=tf.nn.relu, name='conv1')


