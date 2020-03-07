from collections import OrderedDict
import os
import numpy as np


class Robot(object):

    #correct fname got from child class
    def __init__(self, fname):

        self.file = fname
        self.folder = os.path.dirname(fname)


    def get_model(self, mode='mujoco_py'):
        if mode == 'mujoco_py':
            from mujoco_py import load_model_from_path
            # load model
            print('loading model from {}'.format(self.file))
            model = load_model_from_path(self.file)
            return model
        raise ValueError(
            'Unknown model mode'
        )



    @property
    def dof(self):
        raise NotImplementedError


    @property
    def joints(self):
        raise NotImplementedError


    @property
    def init_qpos(self):
        raise NotImplementedError

    



