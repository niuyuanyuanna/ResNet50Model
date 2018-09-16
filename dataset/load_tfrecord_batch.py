#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/2/18 5:16 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import tensorflow as tf
import os

from config.config import config
from models.ResNet import ResNet50


def parse_exmp(serial_exmp):
    feats = tf.parse_single_example(serial_exmp, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(feats['image_raw'], tf.float64)
    image = tf.reshape(image, [64, 64, 3])
    # print(image.shape)
    image = tf.cast(image, tf.float32)
    label = tf.cast(feats['label'], tf.int32)
    label = tf.one_hot(label, depth=190, on_value=1)
    return image, label


def evaluate_op(logits, labels):
    predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(predict, 'float'))


if __name__ == '__main__':
    train_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'train.tfrecords')
    dataset_train = tf.data.TFRecordDataset(train_record_dir)
    dataset_train = dataset_train.map(parse_exmp)
    dataset_train = dataset_train.repeat(2).shuffle(1000).batch(128)
    nBatches = 30577 * 2 / 128
    iter_train = dataset_train.make_one_shot_iterator()

    val_record_dir = os.path.join(config.dataset.trainTFRecord_list[0], 'val.tfrecords')
    dataset_val = tf.data.TFRecordDataset(val_record_dir)
    dataset_val = dataset_val.map(parse_exmp)
    dataset_val = dataset_val.batch(128)
    iter_val = dataset_val.make_one_shot_iterator()

    # handle = tf.placeholder(tf.string, shape=[])
    # iterator = tf.data.Iterator.from_string_handle(handle,
    #                                                dataset_train.output_types,
    #                                                dataset_train.output_shapes)
    train_next_element = iter_train.get_next()
    val_next_element = iter_val.get_next()

    x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name="x")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 190], name="y_")
    x_image = tf.reshape(x, shape=[-1, 64, 64, 3])

    # x, y_ = iterator.get_next()
    # x_image = tf.reshape(x, [-1, 64, 64, 3])
    model = ResNet50()
    logits, training = model.deepnn(x_image)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    optmz = tf.train.AdamOptimizer(1e-4)
    train_op = optmz.minimize(loss)
    eval_op = evaluate_op(logits, y_)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    model_save_path = os.path.join(config.model_path, 'resnet50_model')

    with tf.Session() as sess:
        sess.run(init)
        # handle_train, handle_val = sess.run([x.string_handle() for x in [iter_train, iter_val]])
        for i in range(1000):
            train_img_input, train_img_label = sess.run(train_next_element)
            _, train_loss, train_eval = sess.run([train_op, loss, eval_op],
                                                 feed_dict={x: train_img_input, y_: train_img_label, training: True})
            # _, train_loss, train_eval = sess.run([train_op, loss, eval_op], feed_dict={handle: handle_train})
            print('step %d, train loss: %.5f, acc : %.5f' % (i, train_loss, train_eval))
            if i % 100 == 0 or i == 999:
                saver.save(sess, model_save_path, global_step=i)
                val_img_input, val_img_label = sess.run(val_next_element)
                val_loss, val_eval = sess.run([loss, eval_op],
                                              feed_dict={x: val_img_input, y_: val_img_label, training: False})
                print('step %d ,evaluation  loss : %.5f, acc : %.5f' % (i, val_loss, val_loss))



