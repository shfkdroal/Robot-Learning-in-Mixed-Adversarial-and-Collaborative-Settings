#!/usr/bin/env python
 
import numpy as np
import cv2
import copy
from sys import stdout
import random
# Given image, returns image point and theta to grasp
class Predictors:
    def __init__(self, I, learner=None):
        self.I = I
        self.I_h, self.I_w, self.I_c = self.I.shape
        self.learner = learner
        random.seed(48)
        
    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))
    def to_prob_map(self, fc8_vals):
        sig_scale = 1
        fc8_sig = self.sigmoid_array(sig_scale*fc8_vals)
        fc8_prob_map = fc8_sig/fc8_sig.sum()
        return fc8_prob_map
    def sample_from_map(self, prob_map):
        prob_map_contig = np.ravel(prob_map)
        arg_map_contig = np.array(range(prob_map_contig.size))
        smp = np.random.choice(arg_map_contig, p=prob_map_contig)
        return np.unravel_index(smp,prob_map.shape)
    def random_grasp(self, num_angle=18):
        h_g = np.random.randint(self.I_h)
        w_g = np.random.randint(self.I_w)
        t_g = np.random.randint(num_angle)
        return h_g, w_g, t_g
    def center_grasp(self, num_angle=18):
        h_g = np.int(self.I_h/2)
        w_g = np.int(self.I_w/2)
        t_g = np.int(num_angle/2)
        return h_g, w_g, t_g
    def shakeNet_shake(self, num_actions=15, num_samples=1):
        patch_Is_resized = cv2.resize(self.I,(self.learner.IMAGE_SIZE,self.learner.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        #subtract mean
        patch_Is_resized = (patch_Is_resized - 111).astype(float)/144
        patch_Is_resized = np.expand_dims(patch_Is_resized,axis=0)
        self.probs, self.fc8_vals = self.learner.test_one_instance(patch_Is_resized)
        # Normalizing angle uncertainity
        wf = [0.25, 0.5, 0.25]
        self.fc8_norm_vals = copy.deepcopy(self.fc8_vals)
        # Normalize to probability distribution
        self.fc8_prob_vals = self.to_prob_map(self.fc8_norm_vals)
        # Sample from probability distribution
        self.action_id = self.sample_from_map(self.fc8_prob_vals)
        return self.probs, self.action_id[1]