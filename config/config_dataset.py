#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/1/18 1:46 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import os


def add_dataset_params(config, train_data_list=None, test_data_list=None):

    config.dataset.trainImageDirPath_list = []
    config.dataset.trainImageName_classSymbol_list = []
    config.dataset.testImageDirPath_list = []
    config.dataset.testImageName_list = []
    config.dataset.className_classSymbol_list = []
    config.dataset.id_attribute_list = []
    config.dataset.classSymbol_attributes_list = []
    config.dataset.className_wordEmbeddings_list = []
    config.dataset.trainTFRecord_list = []
    config.dataset.testTFRecord_list = []

    for data_set in train_data_list:
        config.dataset.trainImageDirPath_list.append(
            os.path.join(config.data_path, data_set, 'train'))
        config.dataset.trainImageName_classSymbol_list.append(
            os.path.join(config.data_path, data_set, 'train.txt'))
        config.dataset.className_classSymbol_list.append(
            os.path.join(config.data_path, data_set, 'label_list.txt'))
        config.dataset.id_attribute_list.append(
            os.path.join(config.data_path, data_set, 'attribute_list.txt'))
        config.dataset.classSymbol_attributes_list.append(
            os.path.join(config.data_path, data_set, 'attributes_per_class.txt'))
        config.dataset.className_wordEmbeddings_list.append(
            os.path.join(config.data_path, data_set, 'class_wordembeddings.txt'))
        config.dataset.trainTFRecord_list.append(
            os.path.join(config.data_path, data_set, 'TFRecords')
        )

    for data_set in test_data_list:
        config.dataset.testImageDirPath_list.append(
            os.path.join(config.data_path, data_set, 'test'))
        config.dataset.testImageName_list.append(
            os.path.join(config.data_path, data_set, 'image.txt'))
        config.dataset.id_attribute_list.append(
            os.path.join(config.data_path, data_set, 'attribute_list.txt'))
        config.dataset.classSymbol_attributes_list.append(
            os.path.join(config.data_path, data_set,  'attributes_per_class.txt'))
        config.dataset.className_wordEmbeddings_list.append(
            os.path.join(config.data_path, data_set, 'class_wordembeddings.txt'))
        config.dataset.testTFRecord_list.append(
            os.path.join(config.data_path, data_set, 'TFRecords')
        )
