from collections import OrderedDict
import numpy as np

from grasp.utils import transform_utils as T
from grasp.envs.sawyer import SawyerEnv

from grasp.models.robots import Sawyer

from termcolor import colored

from mujoco_py import functions

class SawyerLift(SawyerEnv):

    def __init__(
            self,
            reward_shaping=False,
            # quite important
            use_object_obs=True,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_collision_mesh=True,
            render_visual_mesh=True,
            control_freq=10,
            horizon=1000,
            ignore_done=False,
            use_camera_obs=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            use_indicator_object=False,
            gripper_type ='LeftTwoFingerGripper',
            use_render=True,
            log_name = '1',
            use_new_model= 'False',
            use_pro_new = 'False',
            to_train='False',
            is_human ='False',
            train_pro =False,
            adv_init=False,
            random_perturb=False,
            use_pro_name='',
            use_new_name='',
            object_xml='',
            user_name='',
            seed = 48,
            params = None,
            test_user = False,
            is_test = False,
            use_filter = False,
            option=0
    ):

        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        super().__init__(
            has_renderer = has_renderer,
            has_offscreen_renderer = has_offscreen_renderer,
            render_visual_mesh = render_visual_mesh,
            render_collision_mesh = render_collision_mesh,
            control_freq = control_freq,
            horizon = horizon,
            ignore_done = ignore_done,
            use_camera_obs = use_camera_obs,
            camera_name = camera_name,
            camera_height = camera_height,
            camera_width = camera_width,
            camera_depth = camera_depth,
            use_indicator_object = use_indicator_object,
            gripper_type = gripper_type,
            use_render = use_render,
            log_name = log_name,
            use_new_model = use_new_model,
            use_pro_new = use_pro_new, 
            to_train = to_train,
            is_human = is_human,
            train_pro = train_pro,
            adv_init=adv_init,
            random_perturb=random_perturb,
            use_pro_name=use_pro_name,
            use_new_name=use_new_name,
            object_xml = object_xml,
            user_name = user_name,
            seed = seed,
            params= params,
            test_user = test_user,
            is_test=is_test,
            use_filter=use_filter,
            option=option
        )



    def _load_model(self):

        super()._load_model()
        # self.mujoco_robot.set_base_xpos([0, 0, 0])
        self.model = self.mujoco_robot



    def _get_reference(self):

        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id('object0')
        self.cube_geom_id = self.sim.model.geom_name2id('object0')


        # debug
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        self.table_body_id = self.sim.model.body_name2id('table0')
        self.cube_geom_id = self.sim.model.geom_name2id('table0')



    def _reset_internal(self):

        super()._reset_internal()
        # init_qpos: start at target_pos1
        idx = []
        idx.extend(self._ref_joint_pos_indexes)
        idx.extend(self._ref_gripper_joint_pos_indexes)
        self.sim.data.qpos[np.array(idx)] = self.mujoco_robot.init_qpos


    # def reward(self, action=None):

    #     '''
    #     Reward function for the task

    #     The dense reward has three components

    #     Reaching: in [0,1], to encourage the arm to reach the cube
    #     Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
    #     Lifting: in {0,1}, non-zero if arm has lifted the cube

    #     The sparse reward only consists of lifting component

    #     :param action:
    #         action (np array): unused for this task
    #     :return:
    #         reward (float): the reward
    #     '''

    #     reward = 0.

    #     # sparse completion reward
    #     if self._check_success():
    #         complete_reward = 1.0
    #     else:
    #         complete_reward = 0.0

    #     reaching_reward =0
    #     grasping_reward =0
    #     # reward shaping
    #     if self.reward_shaping:

    #         #reaching reward
    #         cube_pos = self.sim.data.body_xpos[self.cube_body_id]
    #         gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
    #         dist = np.linalg.norm(gripper_site_pos - cube_pos)
    #         reaching_reward = 1 - np.tanh(10.0 *dist)

    #         #grasping reward

    #         touch_left_finger = False
    #         touch_right_finger = False
    #         # print('contact geoms: ',self.sim.data.contact)
    #         for i in range(len(self.sim.data.contact)):
    #             c = self.sim.data.contact[i]
    #             print('contact geoms 1: {}, 2: {} '.format(self.sim.model.geom_id2name(c.geom1), self.sim.model.geom_id2name(c.geom2)))
    #             if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
    #                 touch_left_finger = True
    #             if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
    #                 touch_left_finger = True
    #             if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
    #                 touch_right_finger = True
    #             if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
    #                 touch_right_finger = True
    #         print('')
    #         if touch_right_finger and touch_left_finger:
    #             grasping_reward = 0.25
    #         else:
    #             grasping_reward =0.0

    #     reward = complete_reward + reaching_reward + grasping_reward
    #     print(colored('total reward: {}, lift reward: {}, reaching reward: {}, grasping reward: {}'.format(reward, complete_reward, reaching_reward, grasping_reward), 'green'))


    #     return reward



    def _adv_check_success(self):
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                touch_left_finger = True
            if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                touch_right_finger = True
            if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        if touch_right_finger and touch_left_finger:
            return False
        else:
            return True



    def _get_observation(self):

        di = super()._get_observation()

        # nomrally set to false
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name = self.camera_name,
                width = self.camera_width,
                height = self.camera_height,
                depth = self.camera_depth,
            )

            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs


        # todo
        # print('sawyer_lift.py: use_obejct_obs', self.use_object_obs)
        if self.use_object_obs:
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            cube_quat = T.convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to='xyzw')
            di['cube_pos'] = cube_pos
            di['cube_quat'] = cube_quat

            # todo
            # gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            # di['gripper_to_cube'] = gripper_site_pos - cube_pos

            # di['object-state'] = np.concatenate(
            #     [cube_pos, cube_quat, di['gripper_to_cube']]
            # )

        return di



    def _check_success(self):

        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        #todo
        table_height = self.sim.data.body_xpos[self.table_body_id][2]

        print('table_height{}, cube_height{}'.format(table_height, cube_height))


        return cube_height > table_height + 0.01





