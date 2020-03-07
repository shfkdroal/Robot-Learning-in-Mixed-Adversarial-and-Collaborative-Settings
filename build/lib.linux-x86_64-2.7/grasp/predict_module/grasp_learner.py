#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from termcolor import colored

import ipdb
from .graspNet import model as grasp_net
# from graspNet import model as grasp_net

class grasp_obj:
    #'C:/Users/stpny/Downloads/grasp_public-master/grasp_public-master/grasp/predict_module/models/checkpoint.ckpt-2000'
    #'./models/shake/checkpoint.ckpt-2000'
    def __init__(self, checkpoint_path='./models/shake/checkpoint.ckpt-2000', gpu_id=-1, num_samples = 128):
        self.checkpoint = checkpoint_path
        if gpu_id==-1:
            self.dev_name = "/cpu:0"
        else:
            self.dev_name = "/gpu:{}".format(gpu_id)

        self.IMAGE_SIZE = 224
        self.NUM_CHANNELS = 3
        self.GRASP_ACTION_SIZE = 18
        self.SEED = 48  # Set to None for random seed.
        self.BATCH_SIZE = num_samples

        #CONFIG PARAMS
        self.INTRA_OP_THREADS = 1
        self.INTER_OP_THREADS = 1
        self.SOFT_PLACEMENT = True

        tf.set_random_seed(self.SEED)

        self.config = tf.ConfigProto(allow_soft_placement=self.SOFT_PLACEMENT,
                intra_op_parallelism_threads=self.INTRA_OP_THREADS,
                inter_op_parallelism_threads=self.INTER_OP_THREADS)
        self.config.gpu_options.allow_growth = True

    def sigmoid_array(self,x):
        return 1 / (1 + np.exp(-x))

    def test_init(self, lr_rate):
        with tf.device(self.dev_name):
            with tf.name_scope('Grasp_training_data'):
                # input
                self.Grasp_patches = tf.placeholder(tf.float32, shape=[None,self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS])
                # groundtruth, debug
                self.y = tf.placeholder(tf.float32, shape=[None, 1])
            with tf.name_scope('Grasp'):
                self.M = grasp_net()
                self.M.initial_weights(weight_file=None)
                self.grasp_pred = self.M.gen_model(self.Grasp_patches)

        with tf.device("/cpu:0"):
            grasp_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Grasp')
            grasp_saver = tf.train.Saver(grasp_variables, max_to_keep=100)



        with tf.name_scope('fc8_norm_vals'):
            # debug, simulate fc8_norm_vals
            padding = tf.constant([[0, 0], [1, 1]])
            grasp_pred = tf.pad(self.grasp_pred, padding, "REFLECT")
            grasp_in = tf.expand_dims(tf.expand_dims(grasp_pred, axis=1), axis=3)  # NHWC [128,1,18,1]
            filter = tf.constant([[0.25, 0.5, 0.25]])  # [1, 3]
            filter = tf.expand_dims(tf.expand_dims(filter, axis=2), axis=3)  # H, W, IN, OUT, [1, 3, 1, 1]
            self.fc8_norm_vals = tf.nn.conv2d(grasp_in, filter, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC') # shape=(?, 1, 18, 1)
            # ipdb.set_trace()
            self.fc8_norm_vals = tf.squeeze(self.fc8_norm_vals, axis=(1,3)) #[128,18] or [None, 18]


        # training op, Jiali
        with tf.device(self.dev_name):
            self.pred = tf.reduce_max(self.fc8_norm_vals, axis=1, keepdims=True) # [None, 1]
            self.probs  = tf.sigmoid(self.pred)
            self.loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.y)
            self.loss = tf.reduce_mean(self.loss_vec)
            optimizer = tf.train.RMSPropOptimizer(lr_rate)
            pro_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Grasp')
            self.train_op = optimizer.minimize(self.loss, var_list=pro_vars)

        # debug, Jiali
        with tf.name_scope('init'):
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        with tf.device(self.dev_name):
            self.sess = tf.Session(config = self.config)
            # add init, Jiali
            self.sess.run(init)
            print(colored('init pro model with: {}'.format(self.checkpoint), 'magenta'))
            grasp_saver.restore(self.sess, self.checkpoint)


    # return fc8
    def test_one_batch(self,Is):
        with tf.device(self.dev_name):
            print('debug test one batch, Grasp_patches: ',Is.shape)
            grasp_feed_dict = {self.Grasp_patches : Is, self.M.dropfc6 : 1.0, self.M.dropfc7 : 1.0}
            pred ,g_pred_norms  = self.sess.run([self.pred, self.fc8_norm_vals], feed_dict=grasp_feed_dict)
        return pred, g_pred_norms

    # training, y_batch: groundtruth, y_pred_batch: predictions
    def train_batch(self, Is, y_batch, save_num, save_name, batch_update):
        n_epochs =1
        n_iteration_per_epoch = 1
        dropfc6 = np.ones((batch_update-1,1))
        dropfc7 = np.ones((batch_update-1, 1))
        train_loss_val = None
        for epoch in range(n_epochs):
            print(colored('Epoch: {}'.format(epoch),'magenta'))
            for iteration in range(n_iteration_per_epoch):
                with tf.device(self.dev_name):
                    train_feed_dict = {self.Grasp_patches: Is, self.y: y_batch, self.M.dropfc6 : dropfc6, self.M.dropfc7 : dropfc7}
                    probs, train_loss_vec, train_loss_val, _ = self.sess.run([self.probs, self.loss_vec, self.loss, self.train_op], feed_dict = train_feed_dict)
                print(colored('probs: {}, pro_train_loss_vec: {}, pro train loss: {}'.format(probs, train_loss_vec,train_loss_val),'magenta'))

        # if save_num % 5 ==0:
        #     self.saver.save(self.sess, save_name)
        #     print(colored('pro model saved at: {}'.format(save_name),'cyan'))
        return train_loss_val


    def test_close(self):
        self.sess.close()
