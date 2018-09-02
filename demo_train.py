#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/1/18 10:32 AM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import pprint
from random import shuffle

from utils.data_analyze import *
from utils.transform import *
from utils.extra_utils import create_logger
from dataset.test_data_loader import TestDataLoader
from dataset.train_data_loader import TrainDataLoader


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

    train_data_loader = TrainDataLoader(train_image_path_list, train_label, config)




