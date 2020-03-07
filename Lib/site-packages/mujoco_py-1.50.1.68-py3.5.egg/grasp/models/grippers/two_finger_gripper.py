import numpy as np
from grasp.models.grippers.gripper import Gripper


class TwoFingerGripperBase(Gripper):
    '''
    Gripper with two fingers
    '''

    def __init__(self):
        super().__init__()



    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])


    @property
    def dof(self):
        return 2


    def contact_geoms(self):
        return [
            'r_finger_g0',
            'r_finger_g1',
            'l_finger_g0',
            'l_finger_g1',
            'r_fingertip_g0',
            'l_fingertip_g0',
        ]


    @property
    def left_finger_geoms(self):
        return ['l_finger_g0', 'l_finger_g1', 'l_fingertip_g0']

    @property
    def right_finger_geoms(self):
        return ['r_finger_g0', 'r_finger_g1', 'r_fingertip_g0']



class TwoFingerGripper(TwoFingerGripperBase):

    @property
    def dof(self):
        return 1



class LeftTwoFingerGripperBase(Gripper):

    def __init__(self):
        super().__init__()



    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])



    @property
    def joints(self):
        return ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint']


    @property
    def dof(self):
        return 2


    # def contact_geoms(self):
    #     return [
    #         'l_g_r_finger_g0',
    #         'l_g_r_finger_g1',
    #         'l_g_l_finger_g0',
    #         'l_g_l_finger_g1',
    #         'l_g_r_fingertip_g0',
    #         'l_g_l_fingertip_g0',
    #     ]

    @property
    def left_finger_geoms(self):
        return ['l_g_l_finger_g0', 'l_g_l_finger_g1', 'l_g_l_fingertip_g0', 'first2', 'xxxx2']


    @property 
    def right_finger_geoms(self):
        return ['l_g_r_finger_g0', 'l_g_r_finger_g1', 'l_g_r_fingertip_g0', 'first3', 'xxxx3']




class LeftTwoFingerGripper(LeftTwoFingerGripperBase):

    @property
    def dof(self):
        return 1