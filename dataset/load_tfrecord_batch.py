#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/2/18 5:16 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import tensorflow as tf

from config.config import config


def read_records(file_path, resize_height, resize_width, type=None):
    filename_queue = tf.train.string_input_producer([file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.unit8)
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)

    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type == 'normalization':
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)
    elif type == 'centralization':
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5

    return tf_image, tf_label


def get_batch_images(images, labels, batch_zise=config.train.batch_size, labels_nums=190, one_hot=False):
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_zise

    images_batch, labels_batch = tf.train.batch([images, labels], batch_size=batch_zise, capacity=capacity)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch







