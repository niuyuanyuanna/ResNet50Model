#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/31/18 9:38 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import logging
import os


def merge_attributes(attribute_list):
    """
    :param attribute_list: path_list
    :return: dict that key is id and value is attribute name.
    EXAMPLE: {1:'is animal'}
    """
    attributes = {}
    if attribute_list is not None:
        for attribute in attribute_list:
            with open(attribute, 'rb') as fid:
                for line in fid:
                    line = line.decode()
                    line = line.strip().split('\t')
                    attribute_id = line[0]
                    attribute_name = line[-1]
                    if attribute_id not in attribute_list:
                        attributes[attribute_id] = attribute_name
                    else:
                        assert attributes[attribute_id] == attribute_name
    logging.info('merge %d file, finally get %d attributes'
                 % (len(attribute_list), len(attributes)))
    return attributes


def merge_attributes_per_class(attributes_per_class_list):
    """
    :param attributes_per_class_list: path_list
    :return: dict that key is abstract class name
     value is a list of score for each attribute
    EXAMPLE: {'ZJL1':[1,0,0,0,0,0,0.5,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]}

    """
    class_attributes = {}
    if attributes_per_class_list is not None:
        for attributes_per_class in attributes_per_class_list:
            with open(attributes_per_class, 'rb') as fid:
                for line in fid:
                    line = line.decode()
                    line = line.strip().split('\t')
                    class_name = line[0]
                    attributes_score_list = [float(score) for score in line[1:]]
                    if class_name not in attributes_per_class:
                        class_attributes[class_name] = attributes_score_list
                    else:
                        assert class_attributes[class_name] == attributes_score_list
    logging.info('merge %d file, finally get %d class attributes' %
                 (len(attributes_per_class_list), len(class_attributes)))
    return class_attributes


def merge_class_wordembeddings(class_wordembeddings_list):
    """
    :param class_wordembeddings_list: path_list
    :return: dict that key is specific class name
     value is a list of score for 300 word vector
    EXAMPLE: {'orange':[-0.24776 -0.12359 0.20986 -0.15834 -0.15827 -0.90116 -0.095702 -0.23005 0.27094 ...]}
    """
    class_wordembeddings = {}
    if class_wordembeddings_list is not None:
        for class_wordembedding in class_wordembeddings_list:
            with open(class_wordembedding, 'rb') as fid:
                for line in fid:
                    line = line.decode()
                    line = line.strip().split(' ')
                    class_name = line[0]
                    score_vector = [float(score) for score in line[1:]]
                    if class_name not in class_wordembeddings:
                        class_wordembeddings[class_name] = score_vector
                    else:
                        assert class_wordembeddings[class_name] == score_vector
    logging.info('merge %d file, finally get %d pair of class name and wordembeddings' %
                 (len(class_wordembeddings_list), len(class_wordembeddings)))
    return class_wordembeddings


def merge_class_symbol_class_name(class_symbol_class_name_list):
    """
    :param class_symbol_class_name_list:
    :return: dict that key is abstract class name
    value is specific class name
    EXAMPLE: {'ZJL240':'dog'}
    """
    label_real_name_dict = {}
    if class_symbol_class_name_list is not None:
        for class_symbol_class_name in class_symbol_class_name_list:
            with open(class_symbol_class_name, 'rb') as fid:
                for line in fid:
                    line = line.decode()
                    line = line.strip().split('\t')
                    abstract_name = line[0]
                    real_name = line[-1]
                    if abstract_name not in label_real_name_dict:
                        label_real_name_dict[abstract_name] = real_name
                    else:
                        assert label_real_name_dict[abstract_name] == real_name
    logging.info('merge %d file, finally get %d real name labels' %
                 (len(class_symbol_class_name_list), len(label_real_name_dict)))
    return label_real_name_dict


def load_image_and_class_symbol(trainImageDirPath_list, trainImageName_classSymbol_list):
    """
    :param trainImageDirPath_list:
    :param trainImageName_classSymbol_list:
    :return:
    """
    assert trainImageDirPath_list is not None
    assert len(trainImageDirPath_list) == len(trainImageName_classSymbol_list)
    image = []
    label = []
    for i, image_path in enumerate(trainImageName_classSymbol_list):
        with open(image_path, 'rb') as fid:
            for line in fid:
                line = line.decode()
                line = line.strip().split('\t')
                if len(line) > 1:
                    label.append(line[1])
                image.append(os.path.join(trainImageDirPath_list[i], line[0]))
    if len(label) == 0:
        return image
    else:
        return image, label