#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from termcolor import colored

import ipdb
# from .graspNet import model as grasp_net
# from graspNet import model as grasp_net

import matplotlib.pyplot as plt
import os
import ast
from grasp.utils.mjcf_utils import root_path_completion3


def conv2d(input, kernel_size, stride, num_filter, name='conv2d'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        # print(input.get_shape())
        filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

        W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b


def conv2d_transpose(input, kernel_size, stride, num_filter, name='conv2d_transpose'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [kernel_size, kernel_size, num_filter, input.get_shape()[3]]
        output_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, num_filter])

        W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d_transpose(input, W, output_shape, stride_shape, padding='SAME') + b


def fc(input, num_output, name='fc'):
    with tf.variable_scope(name):
        num_input = input.get_shape()[1]
        W = tf.get_variable('w', [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [num_output], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, W) + b


def batch_norm(input, is_training):
    out = tf.contrib.layers.batch_norm(input, decay=0.99, center=True, scale=True,
                                       is_training=is_training, updates_collections=None)
    return out


def leaky_relu(input, alpha=0.2):
    return tf.maximum(alpha * input, input)


class force_filter_obj:

    def __init__(self, checkpoint_path='./models/force/checkpoint.ckpt-2000', gpu_id=-1, batch_size=1, Collective_dimension = 2323):
        if gpu_id == -1:
            self.dev_name = "/cpu:0"
        else:
            self.dev_name = "/gpu:{}".format(gpu_id)

        self.checkpoint = checkpoint_path
        self.Collective_dimension = Collective_dimension

        self.Collective_Input = tf.placeholder(tf.float32, [None, self.Collective_dimension, 1, 1], name='Col_in')
        self.ground_truth = tf.placeholder(tf.float32, shape=[None, 1])

        self.SEED = 48  # Set to None for random seed.
        self.BATCH_SIZE = batch_size

        # CONFIG PARAMS
        self.INTRA_OP_THREADS = 1
        self.INTER_OP_THREADS = 1
        self.SOFT_PLACEMENT = True

        self.learning_rate = 1e-5#1e-9  # 1e-4, 20 epoch/ 4e-5, 60 epoch/

        tf.set_random_seed(self.SEED)

        self.config = tf.ConfigProto(allow_soft_placement=self.SOFT_PLACEMENT,
                                     intra_op_parallelism_threads=self.INTRA_OP_THREADS,
                                     inter_op_parallelism_threads=self.INTER_OP_THREADS)
        self.config.gpu_options.allow_growth = True


        self.num_epoch = 5
        self.sess = None
        self.log_step = 1

        self._force_called = False
        self.is_train = tf.placeholder(tf.bool, name='is_training')

        self._init_ops()

    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))

    def _loss1(self, labels, logits):
        # print(labels.get_shape(), logits.get_shape())
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def _loss2(self, labels, logits):
        return tf.reduce_mean(tf.nn.l2_loss(labels - logits))

    def Entropy(self, Input):
        return tf.reduce_mean(-tf.reduce_sum(Input * tf.log(tf.add(Input, self.small_num)), axis=1))

    def Force_Classifier(self, input):
        with tf.variable_scope('Force', reuse=self._force_called):
            self._force_called = True
            # feedInput = tf.concat(input, intention0, axis=1)
            input = tf.tanh(input)
            gen_fc1 = fc(tf.reshape(input, [-1, self.Collective_dimension]), 30, 'fc1') #4 * 4 * 64
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 1, 1, 30])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)

            #gen_conv2 = conv2d(gen_lrelu1, 4, 1, 32, 'conv2')

            #gen_batchnorm2 = batch_norm(gen_conv2, self.is_train)
            #gen_lrelu2 = leaky_relu(gen_batchnorm2)

            #gen_conv3 = conv2d(gen_lrelu2, 4, 1, 16, 'conv3')  # 32 -> 64
            #gen_batchnorm3 = batch_norm(gen_conv3, self.is_train)
            #gen_lrelu3 = leaky_relu(gen_batchnorm3)

            #gen_conv4 = conv2d(gen_lrelu3, 4, 1, 8, 'conv4')  # 3->32
            #gen_batchnorm4 = batch_norm(gen_conv4, self.is_train)  # added by Yoon
            #gen_lrelu4 = leaky_relu(gen_batchnorm4)

            gen_reshape2 = tf.reshape(gen_lrelu1, [-1, 30]) #(gen_lrelu4, [-1, 128]) # 16*64

            #gen_fc2 = fc(gen_reshape2, 6, 'fc1_1')
            #gen_fc2_reshaped = tf.reshape(gen_fc2, [-1, 6])
            #gen_batchnorm3 = batch_norm(gen_fc2_reshaped, self.is_train)
            #p = leaky_relu(gen_batchnorm3)

            #p2 = tf.reshape(p, [-1, 6])
            fc3 = fc(gen_reshape2, 1, 'fc3')
            fc3_rs = tf.reshape(fc3, [-1, 1])
            # temp = tf.sigmoid(gen_fc2_reshaped)
            fc3_rs = batch_norm(fc3_rs, self.is_train)
            is_helpful = tf.sigmoid(fc3_rs)

            return is_helpful  # gen_sigmoid4

    def _session_controle(self, train_samples, is_train):
        ## added by Yoon. 05.08 2019
        with tf.device(self.dev_name):
            self.sess = tf.Session(config=self.config)

            self.Collective_dimension = train_samples[0].shape[0] - 1 #get the dimension of the input

            print(root_path_completion3(self.checkpoint))

            test_bool = False
            if test_bool:#if not os.path.exists(root_path_completion3(self.checkpoint)) and is_train:
                print("pass not exist")
                force_filter = self
                self.sess.run(tf.global_variables_initializer())
                force_filter.train(self.sess, train_samples)
                force_filter_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Force')
                saver = tf.train.Saver(force_filter_var_list)
                saver.save(self.sess, './models/force')
                return
            elif os.path.exists(root_path_completion3(self.checkpoint)) and is_train:
                self.sess = tf.Session(config=self.config)
                self.sess.run(tf.global_variables_initializer())

                force_filter_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Force')
                saver = tf.train.import_meta_graph('./models/force.meta')
                #saver = tf.train.Saver(force_filter_var_list, max_to_keep=100)
                #saver.restore(self.sess, self.checkpoint)
                saver.restore(self.sess, tf.train.latest_checkpoint('./models'))
                self.train(self.sess, train_samples=train_samples)

                print("pass exist- valid check point")
                # ./models/force/checkpoint.ckpt-2000
                return


            # test mode
            self.sess = tf.Session(config=self.config)
            self.sess.run(tf.global_variables_initializer())

            #force_filter_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Force')
            saver = tf.train.import_meta_graph('./models/force.meta')
            # saver = tf.train.Saver(force_filter_var_list, max_to_keep=100)
            # saver.restore(self.sess, self.checkpoint)
            saver.restore(self.sess, tf.train.latest_checkpoint('./models'))
            self.test_output(self.sess, test_samples=train_samples)
        # ./models/force/checkpoint.ckpt-2000


    def test_output(self, sess, test_samples):

        batch_samples = test_samples
        self.BATCH_SIZE = batch_samples.shape[0]
        feed_input = self.Extract_feed_input(batch_samples)

        ground_truth = self.Extract_ground_truth(test_samples)

        force_feed_dict = {self.Collective_Input: feed_input, self.is_train: False}
        output = sess.run([self.helP_or_adv_prediction], feed_dict=force_feed_dict)

        output = np.array(output).reshape(self.BATCH_SIZE, 1)

        for v in range(output.shape[0]):
            if (output[v, 0] >= 0.5 and ground_truth[v, 0]==0) or ((output[v, 0] < 0.5 and ground_truth[v, 0]==1)):
                output[v, 0] = 1
            else:
                output[v, 0] = 0

        #return output
        print("missclassification error: ", np.mean(output, axis=0))



    def _init_ops(self):


        self.helP_or_adv_prediction = self.Force_Classifier(self.Collective_Input)

        force_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "Force")  # 'call' Variables according to the scope predefined

        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        beta = 2e-3
        loss = self._loss1(self.ground_truth, self.helP_or_adv_prediction)
        self.force_loss_op = loss + beta * np.sum(regularizer)
        #self.force_loss_op = self._loss1(self.ground_truth, self.helP_or_adv_prediction)

        #AdamOptimizer(self.learning_rate) #
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)  # According to 'predict_API' init_detector, is 1e-2
        # self.dis_train_op = dis_optimizer.minimize(self.dis_loss_op)

        # print(gen_train_vars)
        # self.gen_train_op = gen_optimizer.minimize(self.gen_loss_op)
        self.force_train_op = optimizer.minimize(self.force_loss_op, var_list=force_train_vars)


    def Extract_ground_truth(self, training_sample):
        ground_truth = training_sample[:, self.Collective_dimension, 0, 0].reshape(self.BATCH_SIZE, 1)
        return ground_truth

    def Extract_feed_input(self, training_sample):
        feed_data_part = training_sample[:, 0:(self.Collective_dimension), 0, 0].\
            reshape(self.BATCH_SIZE, self.Collective_dimension, 1, 1) #1,2324,1,1
        return feed_data_part

    def train(self, sess, train_samples):

        sess.run(tf.global_variables_initializer())
        num_train = train_samples.shape[0]
        step = 0

        # smooth the loss curve so that it does not fluctuate too much
        smooth_factor = 0.95
        plot_dis_s = 0
        plot_gen_s = 0
        plot_ws = 0

        force_losses = []
        max_steps = int(self.num_epoch * (num_train // self.BATCH_SIZE))
        print('Start training ...')
        for epoch in range(self.num_epoch):

            #np.random.shuffle(train_samples)
            for i in range(num_train // self.BATCH_SIZE):
                step += 1

                #np.random.shuffle(train_samples)
                batch_samples = train_samples[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE]

                extracted_ground_truth = self.Extract_ground_truth(batch_samples)
                feed_input = self.Extract_feed_input(batch_samples)

                force_feed_dict = {self.Collective_Input: feed_input, self.ground_truth: extracted_ground_truth,
                                   self.is_train: True}
                _, force_loss = sess.run([self.force_train_op, self.force_loss_op], feed_dict=force_feed_dict)

                plot_gen_s = plot_gen_s * smooth_factor + force_loss * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                force_losses.append(plot_gen_s / plot_ws)

                """
                ValueError: Cannot feed value of shape (1, 2324, 1, 1) for Tensor 'Col_in:0', which has shape '(?, 0, 1, 1)'

                """

                #if step % self.log_step == 0:
                    # print(currentAction)
                print('Iteration {0}/{1}: loss = {2:.4f}'.format(step, max_steps, force_loss))

            plt.plot(force_losses)
            plt.title('force loss')
            plt.xlabel('iterations')
            plt.ylabel('loss')
            # plt.show()
            plt.savefig('force_loss')

        print('... Done!')

    def test_close(self):
        self.sess.close()
