#!/usr/bin/env python
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os
import sys
import time
 
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from IPython import embed
from .shakeNet import model as shake_net
from termcolor import colored
 
class shake_obj:
    def __init__(self, checkpoint_path = './models/initial_models/Shake_model', gpu_id=-1, num_samples=128):
        self.checkpoint = checkpoint_path
        if gpu_id==-1:
            self.dev_name = "/cpu:0"
        else:
            self.dev_name = "/gpu:{}".format(gpu_id)
        self.IMAGE_SIZE = 224
        self.NUM_CHANNELS = 3
        self.SHAKE_FACTOR = 0.25
        self.SHAKE_ACTION_SIZE = 6
        self.SEED = 48  # Set to None for random seed.
        self.BATCH_SIZE = num_samples
        self.TEST_BATCH_SIZE = 1
        self.MAX_STEPS = 10000
 
        #CONFIG PARAMS
        self.INTRA_OP_THREADS = 1
        self.INTER_OP_THREADS = 1
        self.SOFT_PLACEMENT = True
 
        self.LOGDIR = "./logs/reinforce/shake"
        tf.set_random_seed(self.SEED)
         
        self.self_test = False
 
        self.config = tf.ConfigProto(allow_soft_placement=self.SOFT_PLACEMENT,
                intra_op_parallelism_threads=self.INTRA_OP_THREADS,
                inter_op_parallelism_threads=self.INTER_OP_THREADS)
        self.config.gpu_options.allow_growth = True
 

    def sigmoid_array(self,x):
        return 1 / (1 + np.exp(-x))
     
    def test_init(self, adv_lr_rate):
        with tf.device(self.dev_name):
            with tf.name_scope('Shake_training_data'):
                self.Shake_patches = tf.placeholder(tf.float32, shape=[None,self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS])
                #debug
                self.y_adv_gt = tf.placeholder(tf.float32, shape=[None, 1])
            with tf.name_scope('Shake'):
                self.M = shake_net()
                self.M.initial_weights(weight_file=None)
                #debug
                # self.shake_pred = self.M.gen_model(self.Shake_patches)    
        with tf.device("/cpu:0"):
            shake_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Shake')
            shake_saver = tf.train.Saver(shake_variables, max_to_keep=100)
    

        with tf.name_scope('new_layer'):
            #debug, new layer
            drop7 = self.M.gen_model(self.Shake_patches)
            fc8b = tf.Variable(tf.constant(0.1, shape=[self.SHAKE_ACTION_SIZE]))
            fc8W = tf.Variable(tf.truncated_normal([1024, self.SHAKE_ACTION_SIZE], stddev = 0.1))
            self.shake_pred = tf.nn.xw_plus_b(drop7, fc8W, fc8b)
            self.probs = tf.nn.softmax(self.shake_pred, name='y_prob')
            self.y_pred = tf.reduce_max(self.probs, axis =1 , keepdims = True)
        #debug training op
        with tf.name_scope('train'):
            self.loss_vec = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_adv_gt, logits= self.y_pred))
            self.loss = tf.reduce_mean(self.loss_vec)
            optimizer = tf.train.RMSPropOptimizer(adv_lr_rate)
            adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.train_op = optimizer.minimize(self.loss, var_list = adv_vars)

        #debug
        with tf.name_scope('init'):
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        with tf.device(self.dev_name):
            self.sess = tf.Session(config = self.config)
            #debug
            self.sess.run(init)
            shake_saver.restore(self.sess, self.checkpoint)
 
    def test_one_instance(self,Is):
        with tf.device(self.dev_name):
            shake_feed_dict = {self.Shake_patches : Is, self.M.dropfc6 : 1.0, self.M.dropfc7 : 1.0}
            g_probs, g_pred = self.sess.run([self.probs, self.shake_pred], feed_dict=shake_feed_dict)
        return g_probs, g_pred


    #debug
    def train_batch(self, Is, y_batch, batch_update):
        n_epochs = 1
        n_iteration_per_epoch =1 
        dropfc6 = np.ones((batch_update-1, 1))
        dropfc7 = np.ones((batch_update-1, 1))
        train_loss_val = None
        # reverse y_batch
        y_batch = [[1-i[0]] for i in y_batch]
        print(colored('y_batch final: {}'.format(y_batch),'cyan'))
        for epoch in range(n_epochs):
            print(colored('Epoch: {}'.format(epoch), 'cyan'))
            for iteration in range(n_iteration_per_epoch):
                with tf.device(self.dev_name):
                    train_feed_dict = {self.Shake_patches: Is, self.y_adv_gt: y_batch, self.M.dropfc6:dropfc6, self.M.dropfc7: dropfc7}
                    # probs, train_loss_vec, train_loss_val, _ = self.sess.run([self.probs, self.loss_vec, self.loss,  self.train_op])
                    train_loss_val, _ = self.sess.run([self.loss,  self.train_op], feed_dict = train_feed_dict)
                # print(colored('adv_probs: {}, adv_train_loss_vec: {}, adv_train_loss: {}'.format(probs, train_loss_vec, train_loss_val)))
                print(colored('adv_train_loss: {}'.format(train_loss_val), 'cyan'))
        return train_loss_val




 
    def test_close(self):
        self.sess.close()
 
    def train_init(self):
        with tf.device(self.dev_name):
            with tf.name_scope('Shake_training_data'):
                self.Shake_patches = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE,self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS])
                self.Shake_angle = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
                self.Shake_success = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
 
            with tf.name_scope('Shake'):
                self.M = shake_net()
                self.M.initial_weights(weight_file=None)
                self.shake_pred = self.M.gen_model(self.Shake_patches)
            self.shake_loss, self.shake_accuracy = self.M.gen_loss(self.shake_pred, self.Shake_angle, self.Shake_success)
        with tf.device("/cpu:0"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # Step Decay
            shake_learning_rate = tf.train.exponential_decay(
                0.001,                # Base learning rate.
                self.global_step,  # Current index into the dataset.
                100,          # Decay step.
                0.1,                # Decay rate.
                staircase=True,name='shake_learning_rate')
            shake_loss_sum = tf.scalar_summary('shake_loss', self.shake_loss)
            shake_accuracy_sum = tf.scalar_summary('shake_accuracy', self.shake_accuracy)
            shake_lr_sum = tf.scalar_summary('shake_lr', shake_learning_rate)
            shake_step_sum = tf.scalar_summary('shake_step', self.global_step)
            self.shake_summary_op = tf.merge_summary([shake_loss_sum, shake_accuracy_sum, shake_lr_sum, shake_step_sum])
        with tf.device(self.dev_name):
            #Using RMSProp Optimizer
            with tf.name_scope('Optimizer'):
                shake_optimizer = tf.train.RMSPropOptimizer(shake_learning_rate, decay = 0.9, momentum = 0.9)
                self.minimize_shake_net = shake_optimizer.minimize(self.shake_loss, global_step=self.global_step,name='shake_optimizer')
        with tf.device("/cpu:0"):
            shake_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Shake')
            optimizer_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Optimizer')
            variables_to_intialize = optimizer_variables + [self.global_step]
            self.shake_saver = tf.train.Saver(shake_variables, max_to_keep=100)
            init_op = tf.initialize_variables(variables_to_intialize)
        with tf.device(self.dev_name):
            self.sess = tf.Session(config = self.config)
            self.shake_saver.restore(self.sess, self.checkpoint)
            self.sess.run(init_op)
            self.train_writer = tf.train.SummaryWriter(self.LOGDIR,self.sess.graph)
            self.step = 0
 
    def train_one_batch(self,i_shake, theta_labels, success_labels):
          strtime = time.time()
          shake_feed_dict = {self.Shake_patches : i_shake, self.Shake_angle : theta_labels, self.Shake_success : success_labels, self.M.dropfc6 : 0.5, self.M.dropfc7 : 0.5}
          _, g_l, g_a, g_p, shake_summary, self.step, g_sig = self.sess.run([self.minimize_shake_net, self.shake_loss,self.shake_accuracy, self.shake_pred, self.shake_summary_op, self.global_step, self.M.sig_op], feed_dict=shake_feed_dict)
          self.train_writer.add_summary(shake_summary,self.step)
          print('Iter:{} Push Loss:{} PushAccuracy : {} time : {}'.format(self.step,g_l,g_a,time.time()-strtime))
          return g_l,g_a,time.time()-strtime
 
    def train_save(self, fname):
        self.shake_saver.save(self.sess, "{}".format(fname))
 
    def train_close():
        self.sess.close()



