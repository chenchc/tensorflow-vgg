import inspect
import os

import math
import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None, mask_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        if mask_npy_path == None:
            self.mask_dict = {}
            for layer in self.data_dict:
                self.mask_dict[layer] = []
                for idx in self.data_dict[layer]:
                    data = np.ones(idx.shape, dtype=np.bool)
                    self.mask_dict[layer].append(data)

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def pruning_mask(self, bottom, name):
        with tf.variable_scope(name):
            return tf.get_variable('mask', shape=bottom.shape, dtype=tf.Bool, initializer=tf.constant_initializer(True), trainable=False)
        

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        weight = tf.get_variable('filter', shape=self.data_dict[name][0].shape, dtype=tf.float32, initializer=tf.constant_initializer(self.data_dict[name][0]))
        mask = tf.get_variable('mask_filter', shape=self.data_dict[name][0].shape, dtype=tf.bool, initializer=tf.constant_initializer(self.mask_dict[name][0]))
        masked = weight * tf.cast(mask, tf.float32)
        return masked

    def get_bias(self, name):
        weight = tf.get_variable('biases', shape=self.data_dict[name][1].shape, dtype=tf.float32, initializer=tf.constant_initializer(self.data_dict[name][1]))
        mask = tf.get_variable('mask_biases', shape=self.data_dict[name][1].shape, dtype=tf.bool, initializer=tf.constant_initializer(self.mask_dict[name][1]))
        masked = weight * tf.cast(mask, tf.float32)
        return masked

    def get_fc_weight(self, name):
        weight = tf.get_variable('weights', shape=self.data_dict[name][0].shape, dtype=tf.float32, initializer=tf.constant_initializer(self.data_dict[name][0]))
        mask = tf.get_variable('mask_weights', shape=self.data_dict[name][0].shape, dtype=tf.bool, initializer=tf.constant_initializer(self.mask_dict[name][0]))
        masked = weight * tf.cast(mask, tf.float32)
        return masked

    def prune(self, sess, num_iteration=1):
        def prune_layer(name, weight_name, sparsity):
            with tf.variable_scope(name, reuse=True):
                weight = tf.get_variable('{}'.format(weight_name), dtype=tf.float32)
                mask = tf.get_variable('mask_{}'.format(weight_name), dtype=tf.bool)
                weight_nz = tf.boolean_mask(weight, mask)

                values, _ = tf.nn.top_k(tf.abs(weight_nz), k=tf.cast(math.pow(sparsity, 1.0 / num_iteration) * tf.cast(tf.size(weight_nz), tf.float32), tf.int32))
                threshold = tf.gather(values, tf.size(values) - 1)

                update_mask = tf.assign(
                    mask,
                    tf.logical_and(mask, tf.abs(weight) >= threshold))

                sparsity_cur = tf.reduce_mean(tf.cast(mask, tf.float32))

            sess.run(update_mask)
            print '{} sparsity: {}'.format(name, sess.run(sparsity_cur))

        prune_layer('conv1_1', 'filter', 0.58)
        prune_layer('conv1_2', 'filter', 0.22)
        prune_layer('conv2_1', 'filter', 0.34)
        prune_layer('conv2_2', 'filter', 0.36)
        prune_layer('conv3_1', 'filter', 0.53)
        prune_layer('conv3_2', 'filter', 0.24)
        prune_layer('conv3_3', 'filter', 0.42)
        prune_layer('conv4_1', 'filter', 0.32)
        prune_layer('conv4_2', 'filter', 0.27)
        prune_layer('conv4_3', 'filter', 0.34)
        prune_layer('conv5_1', 'filter', 0.35)
        prune_layer('conv5_2', 'filter', 0.29)
        prune_layer('conv5_3', 'filter', 0.36)
        prune_layer('fc6', 'weights', 0.04)
        prune_layer('fc7', 'weights', 0.04)
        prune_layer('fc8', 'weights', 0.23)

    #def retrain
     
if __name__ == '__main__':
    vgg = Vgg16()
