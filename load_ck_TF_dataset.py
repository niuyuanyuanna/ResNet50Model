#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 14:58
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : load_ck_TF_dataset.py
import tensorflow as tf
import numpy as np
import os

from config.config import config
from models.resnet50 import deepnn


def read_records(file_path, resize_height, resize_width, classes):
    filename_queue = tf.train.string_input_producer([file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    print(tf.cast(tf_depth, tf.int32))
    tf_label = tf.cast(features['label'], tf.int32)
    tf_label = tf.one_hot(tf_label, depth=classes, on_value=1)
    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])
    tf_image = tf.cast(tf_image, tf.float32)
    return tf_image, tf_label


def get_batch_images(images, labels, batch_zise=config.train.batch_size, labels_nums=7, one_hot=True):
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_zise

    images_batch, labels_batch = tf.train.batch([images, labels], batch_size=batch_zise, capacity=capacity)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch


def cost(logits, labels):
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    return cross_entropy_loss


def accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_op = tf.reduce_mean(correct_prediction)
    return accuracy_op


# def evaluate(test_features, test_labels, name='test '):
#     tf.reset_default_graph()
#
#     x = tf.placeholder(tf.float32, [None, 64, 64, 3])
#     y_ = tf.placeholder(tf.int64, [None, 6])
#
#     logits, keep_prob, train_mode = deepnn(x)
#     accuracy_step = accuracy(logits, y_)
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, model_save_path)
#         accu = sess.run(accuracy_step, feed_dict={x: test_features, y_: test_labels,
#                                              keep_prob: 1.0, train_mode: False})
#         print('%s accuracy %g' % (name, accu))


def pares_tf(example_proto):
    #调用接口解析一行样本
    features = tf.parse_single_example(serialized=example_proto,
                                             features={
                                                 'image_raw': tf.FixedLenFeature([], tf.string),
                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'depth': tf.FixedLenFeature([], tf.int64),
                                                 'label': tf.FixedLenFeature([], tf.int64)
                                             })
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    print(tf.cast(tf_depth, tf.int32))
    tf_label = tf.cast(features['label'], tf.int32)
    tf_label = tf.one_hot(tf_label, depth=7, on_value=1)
    tf_image = tf.reshape(tf_image, [64, 64, 3])
    tf_image = tf.cast(tf_image, tf.float32)
    return tf_image, tf_label


def load_tf_dataset(file_path):
    dataset = tf.data.TFRecordDataset(filenames=[file_path])
    dataset = dataset.map(pares_tf)
    dataset = dataset.batch(128).repeat(1)

    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    x = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3), name="x")
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="y_")

    image = tf.reshape(x, shape=(-1, 64, 64, 3))
    logits, keep_prob, train_mode = deepnn(x, 7)
    cross_entropy_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)

    # 定义正确值,判断二者下表index是否相等
    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    # 定义如何计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32), name="accuracy")
    # 定义初始化op
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        print("start")
        sess.run(fetches=init)
        i = 0
        try:
            while True:
                # 通过session每次从数据集中取值
                image, label = sess.run(next_element)
                sess.run(train_step, feed_dict={x: image, y_: label})
                if i % 100 == 0:
                    train_accuracy = sess.run(fetches=accuracy, feed_dict={x: image, y_: label})
                    print(i, "accuracy=", train_accuracy)
                i = i + 1
        except tf.errors.OutOfRangeError:
            print("end!")


if __name__ == '__main__':
    file_path = config.dataset.ck.train_TFRecord_file_path
    load_tf_dataset(file_path)



# if __name__ == '__main__':
#     train_file_path = config.dataset.ck.train_TFRecord_file_path
#     val_file_path = config.dataset.ck.val_TFRecord_file_path
#     train_image, train_label = read_records(train_file_path, 64, 64)
#     test_iamge, test_label = read_records(val_file_path, 64, 64)
#
#     X = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='X')
#     Y = tf.placeholder(tf.float32, shape=(None, 7), name='Y')
#
#     logits, keep_prob, train_mode = deepnn(X, 7)
#     cross_entropy_loss = cost(logits, Y)
#
#     with tf.name_scope('adam_optimizer'):
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         with tf.control_dependencies(update_ops):
#             train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)
#
#     graph_location = config.tmp.model_graph
#     if not os.path.exists(graph_location):
#         os.makedirs(graph_location)
#     print('Saving graph to: %s' % graph_location)
#     train_writer = tf.summary.FileWriter(graph_location)
#     train_writer.add_graph(tf.get_default_graph())
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(800):
#             X_mini_batch, Y_mini_batch = get_batch_images(train_image, train_label)
#             train_step.run(feed_dict={X: X_mini_batch, Y: Y_mini_batch,
#                                       keep_prob: 0.5, train_mode: True})
#
#             if i % 20 == 0:
#                 train_cost = sess.run(cross_entropy_loss,
#                                       feed_dict={X: X_mini_batch, Y: Y_mini_batch,
#                                                  keep_prob: 1.0, train_mode: False})
#                 print('step %d, training cost %g' % (i, train_cost))
#
#             if i % 100 == 0:
#                 evaluate(test_iamge, test_label)
#
#         model_save_path = os.path.join(config.model.tmp_tf_model_save_path, 'resnet-50.ckpt')
#         saver.save(sess, model_save_path)



