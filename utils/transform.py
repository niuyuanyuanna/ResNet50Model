#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/1/18 12:00 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import numpy as np

import numpy as np

attributes_name = list(['animal', 'transportation', 'clothes', 'plant', 'tableware', 'device', 'black', 'white',
                        'blue', 'brown', 'orange', 'red', 'green', 'yellow', 'has_feathers', 'has_four_legs',
                        'has_two_legs', 'has_two_arms', 'for_entertainment', 'for_business', 'for_communication',
                        'for_family', 'for_office use', 'for_personal', 'gorgeous', 'simple', 'elegant', 'cute',
                        'pure', 'naive'])


def replace_name_with_symbol(condidate, className_classSymbol_dict):
    new_dict = dict()
    for key in condidate:
        # new_key = className_classSymbol_dict[key]
        real_label_name = list(className_classSymbol_dict.values())
        abstract_label_list = list(className_classSymbol_dict.keys())
        new_key = abstract_label_list[real_label_name.index(key)]
        new_dict[new_key] = condidate[key]
    return new_dict


def make_label(train_class_symbol, classSymbol_attributes_dict, classSymbol_wordEmbeddings_dict):
    label = list()
    for symbol in train_class_symbol:
        item = list()
        attribute_label = classSymbol_attributes_dict[symbol]
        # attribute_label = normalize(np.array(attribute_label))
        item.append(attribute_label)
        item.append(classSymbol_wordEmbeddings_dict[symbol])
        label.append(item)
    return label


def normalize(array):
    array[attributes_name.index('animal'):attributes_name.index('black')] \
        = array[attributes_name.index('animal'):attributes_name.index('black')] >= 1
    array[attributes_name.index('black'):attributes_name.index('has_feathers')] \
        = array[attributes_name.index('black'):attributes_name.index('has_feathers')] >= 0.7
    array[attributes_name.index('has_feathers'):attributes_name.index('for_entertainment')] \
        = array[attributes_name.index('has_feathers'):attributes_name.index('for_entertainment')] >= 1
    array[attributes_name.index('for_entertainment'):attributes_name.index('gorgeous')] \
        = array[attributes_name.index('for_entertainment'):attributes_name.index('gorgeous')] >= 1
    array[attributes_name.index('gorgeous'):attributes_name.index('naive') + 1] \
        = array[attributes_name.index('gorgeous'):attributes_name.index('naive') + 1] >= 1
    return array
