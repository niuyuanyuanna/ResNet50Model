#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/1/18 5:06 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import tensorflow as tf
from random import shuffle
import pprint
import math
import sys

from utils.data_analyze import *
from utils.transform import *
from utils.extra_utils import create_logger
from dataset.test_data_loader import TestDataLoader
from dataset.train_data_loader import TrainDataLoader


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成实数型的属性
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def creat_records(dataset_loader, image_path_list, class_symbol_list, output_record_dir):
    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i in range(len(image_path_list)):
        image = dataset_loader.__getitem__(i)
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _bytes_feature(bytes(class_symbol_list[i], encoding='utf8'))
        }))
        writer.write(example.SerializeToString())
    writer.close


if __name__ == '__main__':
    logger = create_logger(os.path.join(config.log_path, 'train.log'))
    logger.info('config:\n{}'.format(pprint.pformat(config)))

    id_attribute_dict, classSymbol_attributes_dict, className_wordEmbeddings_dict, className_classSymbol_dict, \
    train_class_symbol, train_image_path_list, test_image_path_list = get_all_data(logger)

    if config.sample_test is not None:
        train_class_symbol = train_class_symbol[: config.sample_test]
        train_image_path_list = train_image_path_list[: config.sample_test]

    classSymbol_wordEmbeddings_dict = replace_name_with_symbol(className_wordEmbeddings_dict,
                                                               className_classSymbol_dict)
    train_label = make_label(train_class_symbol, classSymbol_attributes_dict,
                             classSymbol_wordEmbeddings_dict)

    if config.train.split_val is not None:
        temp = list(zip(train_class_symbol, train_image_path_list, train_label))
        shuffle(temp)
        train_class_symbol, train_image_path_list, train_label = zip(*temp)
        len_val = int(len(train_label) * config.train.split_val)

        val_image_path_list = train_image_path_list[: len_val]
        val_class_symbol = train_class_symbol[:len_val]
        train_label = train_label[len_val:]
        train_image_path_list = train_image_path_list[len_val:]
        val_data_loader = TestDataLoader(config, val_image_path_list, val_class_symbol)

        if not os.path.exists(config.dataset.trainTFRecord_list[0]):
            os.makedirs(config.dataset.trainTFRecord_list[0])

        val_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'val.tfrecords')
        creat_records(val_data_loader, val_image_path_list, val_class_symbol, val_record_dir)

    train_data_loader = TrainDataLoader(train_image_path_list, train_label, config)

    if not os.path.exists(config.dataset.trainTFRecord_list[0]):
        os.makedirs(config.dataset.trainTFRecord_list[0])
    train_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'train.tfrecords')
    creat_records(train_data_loader, train_image_path_list, train_class_symbol, train_record_dir)

