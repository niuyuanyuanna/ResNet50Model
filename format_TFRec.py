#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/1/18 5:06 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import tensorflow as tf
import os
import cv2

from config.config import config
from dataset.load_data import get_train_and_val_data
from utils.image_aug import aug_img_func


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def creat_records(image_path_list, class_symbol_list, output_record_dir):
    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i in range(len(image_path_list)):
        image = cv2.imread(image_path_list[i])
        image = aug_img_func(image, config.train.aug_strategy, config)
        image_raw = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(class_symbol_list[i])
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print('write total %d files to tfecords' % len(image_path_list))


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    val_image_path_list, val_class_id, train_image_path_list, train_class_id = get_train_and_val_data()

    make_dir(config.dataset.trainTFRecord_list[0])
    val_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'val.tfrecords')
    creat_records(val_image_path_list, val_class_id, val_record_dir)

    train_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'train.tfrecords')
    creat_records(train_image_path_list, train_class_id, train_record_dir)



