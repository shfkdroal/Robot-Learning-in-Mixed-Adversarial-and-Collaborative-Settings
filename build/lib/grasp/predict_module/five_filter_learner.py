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


def fc(input, num_output, name): #='fc'):
    with tf.variable_scope(name):
        num_input = input.get_shape()[1]
        """
        init_range = 0
        if opt1 == True:
            init_range = np.sqrt((1/6))
        else:
            num_input_explicit = 0
            if opt2 == 0:
                num_input_explicit = 150
            elif opt2 == 1:
                num_input_explicit = 30
            elif opt2 == 2:
                num_input_explicit = 65
            elif opt2 == 3:
                num_input_explicit = 80
            elif opt2 == 4:
                num_input_explicit = 100
            init_range = np.sqrt((2 / num_input_explicit))
        """

        W = tf.get_variable('w', [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02), regularizer=tf.contrib.layers.l2_regularizer(0.001)) #regularizer=tf.contrib.layers.l2_regularizer(0.001)
        b = tf.get_variable('b', [num_output], initializer=tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(0.001))

        return tf.matmul(input, W) + b


def batch_norm(input, is_training):
    out = tf.contrib.layers.batch_norm(input, decay=0.99, center=True, scale=True,
                                       is_training=is_training, updates_collections=None)
    return out


def leaky_relu(input, alpha=0.2):
    return tf.maximum(alpha * input, input)

def relu(input):
    return tf.maximum(0.0, input)

class five_filter_obj:

    def __init__(self, is_alt, opt, lr, checkpoint_path='./models/force/checkpoint.ckpt-2000', gpu_id=-1, batch_size=1, Collective_dimension = 2323):


        if gpu_id == -1:
            self.dev_name = "/cpu:0"
        else:
            self.dev_name = "/gpu:{}".format(gpu_id)


        self.is_alt = is_alt
        self.obj_cat = opt

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

        self.learning_rate = lr#1e-5#1e-9  # 1e-4, 20 epoch/ 4e-5, 60 epoch/
        print("Learning rate: ", self.learning_rate)

        tf.set_random_seed(self.SEED)

        self.config = tf.ConfigProto(allow_soft_placement=self.SOFT_PLACEMENT,
                                     intra_op_parallelism_threads=self.INTRA_OP_THREADS,
                                     inter_op_parallelism_threads=self.INTER_OP_THREADS)
        self.config.gpu_options.allow_growth = True

        self.num_epoch = 40
        self.sess = None
        self.log_step = 1

        self._force_called = False
        self.is_train = tf.placeholder(tf.bool, name='is_training')


        self._init_ops(opt=opt, is_alt=is_alt)

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


    def Force_Classifier1_alt(self, input):
        with tf.variable_scope('Force1_alt', reuse=self._force_called):
            self._force_called = True
            # feedInput = tf.concat(input, intention0, axis=1)
            #input = tf.tanh(input)
            gen_fc1 = fc(tf.reshape(input, [-1, self.Collective_dimension]), 150, 'fc1')  #85
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 1, 1, 150])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)

            gen_reshape2 = tf.reshape(gen_lrelu1, [-1, 150])  # (gen_lrelu4, [-1, 128]) # 16*64

            fc3 = fc(gen_reshape2, 1, 'fc3')
            fc3_rs = tf.reshape(fc3, [-1, 1])
            # temp = tf.sigmoid(gen_fc2_reshaped)
            fc3_rs = batch_norm(fc3_rs, self.is_train)
            #is_helpful = leaky_relu(fc3_rs)
            is_helpful = tf.sigmoid(fc3_rs)
            return is_helpful


    def Force_Classifier2_alt(self, input):
        with tf.variable_scope('Force2_alt', reuse=self._force_called):
            self._force_called = True
            # feedInput = tf.concat(input, intention0, axis=1)
            #input = tf.tanh(input)
            gen_fc1 = fc(tf.reshape(input, [-1, self.Collective_dimension]), 30, 'fc1')  # 4 * 4 * 64
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 1, 1, 30])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)

            gen_reshape2 = tf.reshape(gen_lrelu1, [-1, 30])  # (gen_lrelu4, [-1, 128]) # 16*64

            fc3 = fc(gen_reshape2, 1, 'fc3')
            fc3_rs = tf.reshape(fc3, [-1, 1])
            # temp = tf.sigmoid(gen_fc2_reshaped)
            fc3_rs = batch_norm(fc3_rs, self.is_train)
            is_helpful = tf.sigmoid(fc3_rs)

            return is_helpful  # gen_sigmoid4

    def Force_Classifier3_alt(self, input):
        with tf.variable_scope('Force3_alt', reuse=self._force_called):
            self._force_called = True
            # feedInput = tf.concat(input, intention0, axis=1)
            #input = leaky_relu(input)
            gen_fc1 = fc(tf.reshape(input, [-1, self.Collective_dimension]), 65, 'fc1')  # 4 * 4 * 64
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 1, 1, 65])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)

            gen_reshape2 = tf.reshape(gen_lrelu1, [-1, 65])  # (gen_lrelu4, [-1, 128]) # 16*64   35

            fc3 = fc(gen_reshape2, 1, 'fc3')
            fc3_rs = tf.reshape(fc3, [-1, 1])
            # temp = tf.sigmoid(gen_fc2_reshaped)
            fc3_rs = batch_norm(fc3_rs, self.is_train)
            is_helpful = tf.sigmoid(fc3_rs)

            return is_helpful  # gen_sigmoid4

    def Force_Classifier4_alt(self, input):
        with tf.variable_scope('Force4_alt', reuse=self._force_called):
            self._force_called = True
            # feedInput = tf.concat(input, intention0, axis=1)
            #input = tf.tanh(input)
            gen_fc1 = fc(tf.reshape(input, [-1, self.Collective_dimension]), 80, 'fc1')  # 4 * 4 * 64
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 1, 1, 80])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)

            gen_reshape2 = tf.reshape(gen_lrelu1, [-1, 80])  # (gen_lrelu4, [-1, 128]) # 16*64

            fc3 = fc(gen_reshape2, 1, 'fc3')
            fc3_rs = tf.reshape(fc3, [-1, 1])
            # temp = tf.sigmoid(gen_fc2_reshaped)
            fc3_rs = batch_norm(fc3_rs, self.is_train)
            is_helpful = tf.sigmoid(fc3_rs)

            return is_helpful  # gen_sigmoid4

    def Force_Classifier5_alt(self, input):
        with tf.variable_scope('Force5_alt', reuse=self._force_called):
            self._force_called = True
            # feedInput = tf.concat(input, intention0, axis=1)
            #input = tf.tanh(input)
            gen_fc1 = fc(tf.reshape(input, [-1, self.Collective_dimension]), 100, 'fc1')  # 4 * 4 * 64
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 1, 1, 100])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)

            gen_reshape2 = tf.reshape(gen_lrelu1, [-1, 100])  # (gen_lrelu4, [-1, 128]) # 16*64

            fc3 = fc(gen_reshape2, 1, 'fc3')
            fc3_rs = tf.reshape(fc3, [-1, 1])
            # temp = tf.sigmoid(gen_fc2_reshaped)
            fc3_rs = batch_norm(fc3_rs, self.is_train)
            is_helpful = tf.sigmoid(fc3_rs)

            return is_helpful  # gen_sigmoid4



    def _session_controle(self, train_samples, is_train, should_reuse):
        ## added by Yoon. 05.08 2019


        with tf.device(self.dev_name):
            self.sess = tf.Session(config=self.config)
            tf.reset_default_graph()
            self.Collective_dimension = train_samples[0].shape[0] - 1 #get the dimension of the input

            print(root_path_completion3(self.checkpoint))

            if (not should_reuse or not os.path.exists(root_path_completion3(self.checkpoint))) and is_train:
                print("pass not exist")
                force_filter = self
                self.sess.run(tf.global_variables_initializer())

                force_filter.train(self.sess, train_samples)

                force_filter_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Force')
                saver = tf.train.Saver(force_filter_var_list)


                if self.is_alt:
                    if self.obj_cat == 0:
                        saver.save(self.sess, './models/bottle_alt/force')
                    elif self.obj_cat == 1:
                        saver.save(self.sess, './models/new_cube_alt/force')
                    elif self.obj_cat == 2:
                        saver.save(self.sess, './models/cube_alt/force')
                    elif self.obj_cat == 3:
                        saver.save(self.sess, './models/half-nut_alt/force')
                    elif self.obj_cat == 4:
                        saver.save(self.sess, './models/round-nut_alt/force')
                else:
                    if self.obj_cat == 0:
                        saver.save(self.sess, './models/bottle/force')
                    elif self.obj_cat == 1:
                        saver.save(self.sess, './models/new_cube/force')
                    elif self.obj_cat == 2:
                        saver.save(self.sess, './models/cube/force')
                    elif self.obj_cat == 3:
                        saver.save(self.sess, './models/half-nut/force')
                    elif self.obj_cat == 4:
                        saver.save(self.sess, './models/round-nut/force')
                #self.sess.close()
                return "Mode-fresh-train"
            elif os.path.exists(root_path_completion3(self.checkpoint)) and is_train:
                self._load_controle()
                self.train(self.sess, train_samples=train_samples)

                print("pass exist- valid check point")
                # ./models/force/checkpoint.ckpt-2000
                #self.sess.close()
                return "Mode-reuse-train"


            # test mode
            print("test miss classification error . . .")
            #self.sess.run(tf.global_variables_initializer())
            graph = self._load_controle()
            return self.test_output(self.sess, test_samples=train_samples, graph=graph)
        # ./models/force/checkpoint.ckpt-2000

    def _load_controle(self):

        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        meta_graph_path = None
        if self.is_alt:
            if self.obj_cat == 0:
                meta_graph_path = './models/bottle_alt'
            elif self.obj_cat == 1:
                meta_graph_path = './models/new_cube_alt'
            elif self.obj_cat == 2:
                meta_graph_path = './models/cube_alt'
            elif self.obj_cat == 3:
                meta_graph_path = './models/half-nut_alt'
            elif self.obj_cat == 4:
                meta_graph_path = './models/round-nut_alt'
        else:
            if self.obj_cat == 0:
                meta_graph_path = './models/bottle'
            elif self.obj_cat == 1:
                meta_graph_path = './models/new_cube'
            elif self.obj_cat == 2:
                meta_graph_path = './models/cube'
            elif self.obj_cat == 3:
                meta_graph_path = './models/half-nut'
            elif self.obj_cat == 4:
                meta_graph_path = './models/round-nut'

        loader = tf.train.import_meta_graph((meta_graph_path + '/force.meta'))
        print("meta path: ", meta_graph_path + '/force.meta')
        loader.restore(self.sess, tf.train.latest_checkpoint(meta_graph_path))

        #print(self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Force")))
        graph = tf.get_default_graph()
        return graph


    def test_output(self, sess, test_samples, graph):


        loaded_helP_or_adv_prediction = graph.get_tensor_by_name("adv_pred:0")
        Collective_Input = graph.get_tensor_by_name("Col_in:0")
        is_train = graph.get_tensor_by_name("is_training:0")

        #print(self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Force")))
        #init = tf.local_variables_initializer() #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             #"Force")
        #sess.run(init)

        batch_samples = test_samples
        self.BATCH_SIZE = batch_samples.shape[0]
        feed_input = self.Extract_feed_input(batch_samples)

        ground_truth = self.Extract_ground_truth(test_samples)

        force_feed_dict = {Collective_Input: feed_input, is_train: False}
        output = sess.run([loaded_helP_or_adv_prediction], feed_dict=force_feed_dict)

        output = np.array(output).reshape(self.BATCH_SIZE, 1)

        #print(output)


        """
        for v in range(output.shape[0]):
            if (output[v, 0] >= 0.5 and ground_truth[v, 0]==0) or ((output[v, 0] < 0.5 and ground_truth[v, 0]==1)):
                output[v, 0] = 1
            else:
                output[v, 0] = 0

        #return output
        return ("missclassification error: " + str(np.mean(output, axis=0)))

        """

        print("output: ", output[0])
        print("obj cat: ", self.obj_cat)


        for v in range(output.shape[0]):
            if output[v, 0] >= 0.5:
                output[v, 0] = 1
            else:
                output[v, 0] = 0

        return output










    def _init_ops(self, opt, is_alt):


        self.helP_or_adv_prediction = None
        if not is_alt:
            if opt == 0:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier1(self.Collective_Input), name='adv_pred')
            elif opt == 1:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier2(self.Collective_Input), name='adv_pred')
            elif opt == 2:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier3(self.Collective_Input), name='adv_pred')
            elif opt == 3:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier4(self.Collective_Input), name='adv_pred')
            elif opt == 4:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier5(self.Collective_Input), name='adv_pred')
        else:
            if opt == 0:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier1_alt(self.Collective_Input), name='adv_pred')
            elif opt == 1:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier2_alt(self.Collective_Input), name='adv_pred')
            elif opt == 2:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier3_alt(self.Collective_Input), name='adv_pred')
            elif opt == 3:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier4_alt(self.Collective_Input), name='adv_pred')
            elif opt == 4:
                self.helP_or_adv_prediction = tf.identity(self.Force_Classifier5_alt(self.Collective_Input), name='adv_pred')


        force_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "Force")  # 'call' Variables according to the scope predefined

        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        beta = 1e-12
        if self.obj_cat == 0:
            beta = 0#1e-7#5 #5#4#8#1.3e-8
            self.learning_rate = 3e-3 #5e-4
            self.num_epoch = 300#300#500#350#350
            self.BATCH_SIZE = 100#45

        elif self.obj_cat == 1:
            beta = 0#1e-3
            self.learning_rate = 3e-3
            self.num_epoch = 300
            self.BATCH_SIZE = 100#45

        elif self.obj_cat == 2:
            #beta = 1e-12
            beta = 0#1e-2 # 3e-10
            #self.BATCH_SIZE = 10
            self.learning_rate = 3e-3  # 1e-54
            self.num_epoch = 300
            self.BATCH_SIZE = 100

        elif self.obj_cat == 3:
            beta = 0#1.5e-5#5e-8#3e-10#1e-11 #shoud increase
            self.learning_rate = 5e-4#5#5e-6
            self.num_epoch = 300
            self.BATCH_SIZE = 100

        elif self.obj_cat == 4:
            beta = 0#1e-10 #3e-10
            self.learning_rate = 5e-3#3e-3
            self.num_epoch = 500
            self.BATCH_SIZE = 100


        #loss = self._loss2(self.ground_truth, self.helP_or_adv_prediction)
        loss = self._loss2(self.ground_truth, self.helP_or_adv_prediction)
        #loss = loss/2
        self.force_loss_op = loss + beta * np.sum(regularizer)
        #self.force_loss_op = self._loss1(self.ground_truth, self.helP_or_adv_prediction)

        #optimizer = tf.train.AdamOptimizer(self.learning_rate) #
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

            if self.obj_cat != 0 or self.obj_cat != 3 or self.obj_cat != 4 or self.obj_cat != 2:
                np.random.shuffle(train_samples)

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
