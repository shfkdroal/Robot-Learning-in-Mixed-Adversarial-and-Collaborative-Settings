'''
Be careful about action_spec defined here
'''

from collections import OrderedDict
import numpy as np

from grasp.envs import MujocoEnv
from grasp.models.robots import Sawyer
from grasp.utils import transform_utils as T

from grasp.models.grippers import gripper_factory

from termcolor import colored




class SawyerEnv(MujocoEnv):

    def __init__(
            self,
            has_renderer=False,
            has_offscreen_renderer= False,
            render_collision_mesh= True,
            render_visual_mesh= True,
            control_freq=10,
            horizon=1000,
            ignore_done=False,
            use_camera_obs = False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            use_indicator_object=False,
            gripper_type = None,
            use_render=True,
            log_name = '1',
            use_new_model = 'False',
            use_pro_new='False',
            to_train='False',
            is_human ='False',
            train_pro=False,
            adv_init=False,
            random_perturb=False,
            use_pro_name='',
            use_new_name='',
            object_xml='',
            user_name='',
            seed = 48,
            params = None,
            test_user = False,

    ):

        self.use_indicator_object = use_indicator_object
        self.gripper_type = gripper_type



        super().__init__(
            has_renderer = has_renderer,
            has_offscreen_renderer= has_offscreen_renderer,
            render_collision_mesh= render_collision_mesh,
            render_visual_mesh = render_visual_mesh,
            control_freq = control_freq,
            horizon = horizon,
            ignore_done = ignore_done,
            use_camera_obs = use_camera_obs,
            camera_name = camera_name,
            camera_height = camera_height,
            camera_width = camera_width,
            camera_depth = camera_depth,
            use_render = use_render,
            log_name = log_name,
            use_new_model = use_new_model,
            use_pro_new = use_pro_new,
            to_train=to_train,
            is_human = is_human,
            train_pro = train_pro,
            adv_init=adv_init,
            random_perturb=random_perturb,
            use_pro_name=use_pro_name,
            use_new_name = use_new_name,
            object_xml = object_xml,
            user_name = user_name,
            seed = seed, 
            params = params,
            test_user = test_user,
        )


    def _load_model(self):

        super()._load_model()
        self.mujoco_robot = Sawyer()
        # debug
        self.gripper = gripper_factory(self.gripper_type)()



    def _reset_internal(self):

        super()._reset_internal()



    def _get_reference(self):

        super()._get_reference()

        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes =[
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]

        self._ref_joint_vel_indexes =[
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr('pos_indicator')
            self._ref_indicator_pos_low , self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model_get_joint_qvel_addr('pos_indicator')
            self._ref_indicator_vel_low , self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id('pos_indicator')


        #  [], used in _pre_action
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith('pos')
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith('vel')
        ]


        # debug
        self.gripper_joints = list(self.mujoco_robot.gripper_joints)
        self._ref_gripper_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
        ]

        self._ref_gripper_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
        ]


        # to check
        # self.eef_site_id = self.sim.model.site_name2id('grip')



    # _pre_action is not used by step_IK
    def _pre_action(self, action):

        assert len(action) == self.dof, 'environment got invalid action dimension'
        low, high = self.action_spec
        action = np.clip(action, low, high)


        # rescale normalized action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange
        bias = 0.5 * (ctrl_range[:,1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_action = bias + weight * action
        self.sim.data.ctrl[:] = applied_action

        # todo
        # # gravity compensation
        # self.sim.data.qfrc_applied[
        #     self._ref_joint_vel_actuator_indexes
        # ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]




    def _post_action(self, action):

        ret = super()._post_action(action)
        return ret




    def _get_observation(self):

        di = super()._get_observation()

        di['joint_pos'] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )

        di['joint_vel'] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states =[
            np.sin(di['joint_pos']),
            np.cos(di['joint_pos']),
            di['joint_vel'],
        ]

        # if self.has_gripper:
        #     di['gripper_qpos'] = np.array(
        #         [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
        #     )
        #
        #     robot_states.extend([di['gripper_qpos']])

        # flatten
        di['robot-state'] = np.concatenate(robot_states)
        return di


    '''
    grasp_pt [2x1]
    quat [4x1]
    '''
    @property
    def action_spec(self):

        # in terms of joints

        # low = np.ones(self.dof) * -1.
        # high = np.ones(self.dof) * 1.
        # return low, high

        # in terms of (x,y,quaternion)
        low =  np.ones(6) * -1.
        high = np.ones(6) * 1.
        return low, high




    @property
    def dof(self):

        dof = self.mujoco_robot.dof
        return dof



    @property
    def ref_joint_pos_indexes(self):
        return self._ref_joint_pos_indexes


    @property
    def ref_gripper_pos_indexes(self):
        return self._ref_gripper_joint_pos_indexes




    def pose_in_base_from_name(self, name):

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3,3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3,3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_base_in_world = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_base_in_world)
        return pose_in_base



    def set_robot_joint_positions(self, pos):
        self.sim.data.qpos[self._ref_joint_pos_indexes] = pos
        self.sim.forward()




    @property
    def _right_hand_joint_cartesian_pose(self):
        return self.pose_in_base_from_name("left_gripper_base")


    @property
    def _right_hand_pose(self):
        return self.pose_in_base_from_name("left_gripper_base")


    @property
    def _right_hand_quat(self):
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

