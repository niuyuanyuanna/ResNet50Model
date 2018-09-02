#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/2 18:00
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import os
import tensorflow as tf

from config.config import config
from models.ResNet50 import ResNet50
from dataset.load_tfrecord_batch import *


def save_model(sess, step):
    save_path = config.model_path
    model_name = 'model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_path, model_name), global_step=step)


def get_acc(logits, label):
    current = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), 'float')
    accuracy = tf.reduce_mean(current)
    return accuracy


def train(config):
    train_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'train.tfrecords')
    val_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'val.tfrecords')
    images, labels = read_records(train_record_dir, [64, 64, 3])
    images_batch, labels_batch = get_batch_images(images, labels)

    inputs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='inputs')
    labels = tf.placeholder(tf.string, shape=[None], name='labels')
    drop_prob = tf.placeholder('float')

    resnet_model = ResNet50('channels_last')
    logits = resnet_model.call(inputs)

    acc = get_acc(logits, labels)    # one-hot编码
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    global_step = tf.Variable(0, name='global_step')
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy, global_step=global_step)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(1000):
        images_train, label_train = sess.run([images_batch, labels_batch])
        if i % 100 == 0:
            accuracy = sess.run(acc, feed_dict={x:images_train, y:label_train})






if __name__ == '__main__':



