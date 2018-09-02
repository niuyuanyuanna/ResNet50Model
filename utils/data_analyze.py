#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/1/18 10:38 AM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
from config.config import config
from dataset.load_data import *


def get_all_data(logger):
    id_attribute_dict = merge_attributes(config.dataset.id_attribute_list)
    classSymbol_attributes_dict = merge_attributes_per_class(config.dataset.classSymbol_attributes_list)
    className_wordEmbeddings_dict = merge_class_wordembeddings(config.dataset.className_wordEmbeddings_list)
    className_classSymbol_dict = merge_class_symbol_class_name(config.dataset.className_classSymbol_list)

    # load data and label for train and test
    train_image_path_list, train_class_symbol = load_image_and_class_symbol(
        config.dataset.trainImageDirPath_list, config.dataset.trainImageName_classSymbol_list)

    logging.info("load train data from {} data set, total item {}"
                 .format(len(config.dataset.trainImageDirPath_list), len(train_image_path_list)))
    logger.info('{} classes in train data, total {}, so there are {} classes not exist in train data set'
                .format(len(set(train_class_symbol)), len(classSymbol_attributes_dict),
                        len(classSymbol_attributes_dict)-len(set(train_class_symbol))))

    test_image_path_list = load_image_and_class_symbol(config.dataset.testImageDirPath_list,
                                                       config.dataset.testImageName_list)
    logger.info('load test data from %d data set, total item %d'
                % (len(config.dataset.testImageDirPath_list), len(test_image_path_list)))

    return id_attribute_dict, classSymbol_attributes_dict, className_wordEmbeddings_dict, \
        className_classSymbol_dict, train_class_symbol, train_image_path_list, test_image_path_list

