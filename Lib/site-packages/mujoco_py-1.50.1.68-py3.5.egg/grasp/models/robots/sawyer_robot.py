import numpy as np
from grasp.models.robots.robot import Robot
from grasp.utils.mjcf_utils import xml_path_completion

class Sawyer(Robot):

    def __init__(self):
        super().__init__(xml_path_completion("Baxter/baxter/master.xml"))


    def set_base_xpos(self, pos):
        pass


    @property
    def dof(self):
        # 14 + 4 from grippers
        return 18

    @property
    def joints(self):
        out = []
        for s in ['right_', 'left_']:
            out.extend((s + a for a in ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]))


        return out


    @property
    def gripper_joints(self):
        out = []
        for s in ['r_gripper_', 'l_gripper_']:
            out.extend((s + a for a in ['l_finger_joint', 'r_finger_joint']))
        return out


    @property
    def init_qpos(self):
        # joint positions
        # normal_joint = np.array([
        #     0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297,
        #     -0.518, -0.026, -0.076, 0.175, -0.748, 1.641, -0.158])

        joints = np.array([
            0., 0., 0., 0., 0., 0., 0.,
            -0.98473254,  0.05363101,  0.97193983, -0.57413665, -1.02427219, 1.83454327,  0.88820527,
            0.,0.,0.,0.
        ])
        return joints


    @property
    def gripper_init_qpos(self):
        return np.array([0, 0, 0, 0])