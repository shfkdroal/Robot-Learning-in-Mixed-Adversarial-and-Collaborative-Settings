from collections import OrderedDict
from mujoco_py import MjSim, MjViewer
from mujoco_py import load_model_from_xml, load_model_from_path
import mujoco_py.cymj as cymj
from mujoco_py.generated import const
from mujoco_py import functions

from grasp.utils import MujocoPyRenderer
from grasp.controllers import inverse_kinematics
from grasp.utils.mjcf_utils import xml_path_completion
from grasp.utils.mjcf_utils import image_path_completion
from grasp.utils.mjcf_utils import log_path_completion
from grasp.utils.mjcf_utils import human_path_completion
from grasp.utils.mjcf_utils import model_path_completion
from grasp.utils.mjcf_utils import loss_path_completion
from grasp.utils.mjcf_utils import config_path_completion
from grasp.utils.mjcf_utils import preprocess
from grasp.utils.mjcf_utils import rotateImageAndExtractPatch


from grasp.predict_module import predict_from_img
from grasp.predict_module.predict_API import predict_from_R_table
from grasp.predict_module import init_detector
from grasp.predict_module.predict_API import init_detector_test
from grasp.predict_module.predict_API import init_detector_test_use_filter
from grasp.predict_module.predict_API import init_detector_with_filter

from grasp.predict_module import build_network
from grasp.predict_module import train_adv
from grasp.predict_module import prepare_X_batch
from grasp.predict_module import prepare_X_batch2
# debug
from grasp.predict_module import adv_predict_from_img
from grasp.predict_module import init_adv
from grasp.predict_module.predict_API import init_force_filter2


import os
import time
import math
import numpy as np
import imageio
import imutils
from termcolor import colored
import random
from time import gmtime, strftime
import xml.etree.ElementTree as ET
import glfw
import shutil
import gc
import copy
from time import sleep
import ipdb
import ast


REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


'''
will call init -> base.reset_internal() -> load_model
'''
def make(env_name, *args, **kwargs):
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )

    print('Registered_ENV: {}'.format(REGISTERED_ENVS))
    return REGISTERED_ENVS[env_name](*args, **kwargs)



class EnvMeta(type):

    def __new__(meta, name, base, class_dict):
        cls = super().__new__(meta, name, base, class_dict)

        _unregistered_envs = ['MujocoEnv','SawyerEnv']

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls





class MujocoEnv(metaclass = EnvMeta):

    def __init__(
            self,
            has_renderer=True,
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
            use_render = True,
            log_name = '1',
            use_new_model = 'False',
            use_pro_new='False',
            to_train ='False',
            is_human ='False',
            train_pro='False',
            adv_init=False,
            random_perturb=False,
            use_pro_name='',
            use_new_name='',
            object_xml='',
            user_name='',
            seed =48,
            params =None,
            test_user = False,
            is_test=False,
            use_filter=False,
            option=0

    ):
        self.has_renderer = True
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.viewer = None
        self.model = None
        self.use_new_name=use_new_name
        self.use_pro_name = use_pro_name
        self.user_name = user_name
        self.params = params
        self.begin_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())



        #random init of object size, location and orientation
        if object_xml =='all.xml':
            self.random_init= True
        else:
            self.random_init = False

        self.object_xml = object_xml
        print('model used: {}'.format(self.object_xml))

        # settings for camera observations
        self.use_camera_obs = use_camera_obs
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Camera observations require an offscreen renderer.")
        self.camera_name = camera_name
        if self.use_camera_obs and self.camera_name is None:
            raise ValueError("Must specify camera name when using camera obs")
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth

        # required by stable_baselines monitor
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

        self.speed = 1.
        self.pert = 1
        self._reset_internal()

        #initialize detector only once
        self.num_samples = 128
        self.batch_update = 2
        self.train_batch_update= 5#7

        self.record_state_force_filter = []

        print(colored('initialize protagonist detection model', 'red'))
        self.lr_rate =0.001


        #self.random_init = True ### test script
        ###added by Yoon for test
        #self.object_xml = 'bottle.xml'
        self.adv_start_idx = 100
        self.is_valid_sample = False
        self.use_force_filter = use_filter#False
        self.is_test = is_test#False
        self.contained_curriculum = None
        self.contained_curriculum_test = None
        self.option = option
        self.empty_num = 0
        self.lift_success = False
        self.failed_cases_stack1 = []
        self.failed_cases_stack2 = []
        self.max_steps = 0
        self.is_data_collect_mode = False

        self.num_help_samples = 0
        self.num_adv_samples = 0

        self.patch_Is_resized_vec2=[]
        self.y_train_vec2=[]
        self.fc8_predictions_vec2=[]
        self.pro_save_num2 = 1

        if not self.random_init:
            if self.object_xml =='half-nut.xml' :
                self.lr_rate = 0.00001
            if self.object_xml=='round-nut.xml':
                self.lr = 0.005

        #Modified by Yoon
        self.G = None
        self.G_filter = None
        self.G_force = None
        if not is_test:
           self.G = init_detector(self.num_samples, use_pro_new, use_pro_name, self.num_samples,
                               gpu_id=0, lr_rate=self.lr_rate, test_user=test_user)
           self.G_filter = init_detector_with_filter(self.num_samples, use_pro_new, use_pro_name, self.num_samples,
                               gpu_id=0, lr_rate=self.lr_rate, test_user=test_user)
        elif use_filter:
            self.G = init_detector_test_use_filter(self.num_samples, use_pro_new, use_pro_name, self.num_samples,
                                                   gpu_id=0, lr_rate=self.lr_rate, test_user=test_user, option=self.option)

        elif not use_filter:
            self.G = init_detector_test(self.num_samples, use_pro_new, use_pro_name, self.num_samples,
                                                   gpu_id=0, lr_rate=self.lr_rate, test_user=test_user, option=self.option)
        self.is_once_created_R_table = False
        self.is_once_created_filter = False
        self.R_table = None
        #self.F_table = None
        self.crop_h_ = None
        self.crop_w_ = None
        self.min_coord = None
        self.range_ = None
        self.should_load_R_table = False
        self.training_R_table_ground_truth = False
        self.count_to_100 = 0
        self.current_coord_offset_X = 0#200#60#270
        self.current_coord_offset_Y = 0
        self.post_reward_list = []
        self.stop_training = False


        # the first means no force is applied
        # object to use
        force = 3.5
        if not self.random_init:
            force = 3.5
            if self.object_xml=='bottle.xml':
                force= 3.5
            if self.object_xml=='bread.xml':
                force=3
            if self.object_xml=='new_cube.xml':
                force=3.0
            if self.object_xml=='round-nut.xml':
                force=6 #later changed to 4
            if self.object_xml =='half-nut.xml':
                force = 5.5

            if self.object_xml =='cube.xml':
                force = 1.5

        down_force = 0.3
        if not self.random_init:
            down_force = 1
            if self.object_xml == 'cube.xml':
                down_force = 0.3
            if self.object_xml == 'new_cube.xml':
                down_force = 0.5
            if self.object_xml =='half-nut.xml':
                down_force= 0.3

        #force *= 1.18 #modified by Yoon

        print(colored('obejct_xml: {}, force: {}, down_force: {}'.format(self.object_xml, force, down_force),'red'))

        self.is_human = is_human
        if self.is_human:
            self.adv_forces = np.array([[0, 0, 0, 0, 0, 0],[-force, 0, 0, 0, 0, 0], [0, -force, 0, 0, 0, 0], [force ,0, 0, 0, 0, 0], [0, force, 0, 0, 0, 0], [0, 0, force, 0, 0, 0], [0, 0, -down_force, 0, 0, 0]])
            self.idx_to_action = {0: 'nothing', 1: 'left', 2: 'outward',  3: 'right', 4: 'inside', 5: 'up', 6:'down'}
        else:
            self.adv_forces = np.array([[-force, 0, 0, 0, 0, 0] , [0 , 0, -down_force, 0, 0, 0], [0, -force, 0, 0, 0, 0], [force ,0, 0, 0, 0, 0], [0, force, 0, 0, 0, 0], [0, 0, force, 0, 0, 0]])
            # self.adv_forces = np.array([[0, -force, 0, 0, 0, 0]])
            self.idx_to_action = {0:'left', 1: 'down', 2:'outward', 3:'right', 4:'inside', 5:'up'}
            # self.idx_to_action = {0:'outward'}

        self.n_adv_outputs = len(self.adv_forces)
        self.log_name = log_name

        # init adv policy, get sess
        self.adv_init=adv_init
        self.G_adv = None
        if not self.is_human and self.adv_init:
            print(colored('initialize adversarial detection model', 'blue'))
            # self.adv_policy = build_network(outputs=self.n_adv_outputs, use_new_model=use_new_model, use_new_name=self.use_new_name, log_name=self.log_name, gpu_id=3)
            self.G_adv = init_adv(self.num_samples, use_new_model, use_new_name, self.num_samples, gpu_id=0, adv_lr_rate =0.001)

        #self.G_force = init_force_filter(batch_size=1, gpu_id=0, lr_rate=0.01, Collective_dimension=2323)
        self.fc8_norms = None

        # collect adv inputs
        self.rot_images=[]
        self.rot_y_batches=[]
        self.train_nums =1
        self.pro_train_nums=1
        self.pro_train_nums_filter=1
        self.save_num =1
        self.pro_save_num =1


        # record reward, adv_error
        self.adv_error_logs=[]
        self.reward_logs=[]

        self.total_steps = 0

        self.use_render = use_render
        self.use_new_model = use_new_model
        self.to_train = to_train
        self.intention = 0#-1 #initial intention == help

        self.error_log_path = log_path_completion('error_log{}.txt'.format(self.log_name))
        self.reward_log_path = log_path_completion('reward_log{}.txt'.format(self.log_name))
        self.adv_loss_log_path = loss_path_completion('adv_loss_log{}.txt'.format(self.log_name))
        self.pro_loss_log_path = loss_path_completion('pro_loss_log{}.txt'.format(self.log_name))
        self.pro_loss_log_path2 = loss_path_completion('pro_loss_log_filter{}.txt'.format(self.log_name))
        self.config_log_path = config_path_completion('config_log{}.txt'.format(self.log_name))

        # if file already exists, delete first
        if os.path.exists(self.error_log_path):
            os.remove(self.error_log_path)
        if os.path.exists(self.reward_log_path):
            os.remove(self.reward_log_path)
        if os.path.exists(self.adv_loss_log_path):
            os.remove(self.adv_loss_log_path)
        if os.path.exists(self.pro_loss_log_path):
            os.remove(self.pro_loss_log_path)
        if os.path.exists(self.config_log_path):
            os.remove(self.config_log_path)

        self.human_error_log_path = human_path_completion('human_error_log{}.txt'.format(self.log_name))
        self.human_reward_log_path = human_path_completion('human_reward_log{}.txt'.format(self.log_name))

        if os.path.exists(self.human_error_log_path):
            os.remove(self.human_error_log_path)
        if os.path.exists(self.human_reward_log_path):
            os.remove(self.human_reward_log_path)

        # delete predict image directory is already exists
        if os.path.exists(image_path_completion(self.log_name)):
            shutil.rmtree(image_path_completion(self.log_name))


        # train protagonist policy
        self.train_pro = train_pro
        self.patch_Is_resized_vec=[]
        self.y_train_vec=[]
        self.fc8_predictions_vec=[]


        # inference
        self.infer_adv = False
        self.random_perturb= random_perturb


        #log
        self.write_log_num =1
        self.write_log_update = 5

        # alpha 
        self.alpha = 0.20
        if not self.random_init:
            self.alpha= 0.20
            if self.object_xml == 'cube.xml':
                self.alpha = 0.25
            elif self.object_xml =='bottle.xml':
                self.alpha = 0.20
            elif self.object_xml == 'half-nut.xml':
                self.alpha = 0.5
        self.early_stop_num = 45
        self.seed = seed

        #fix random perturb seed
        np.random.seed(int(time.time() + np.random.randint(10, 50)))#np.random.randint(10, 50)) #modified by Yoon #self.seed)



        # write log info
        print(colored('print logging info ', 'red'))
        with open(self.config_log_path, 'w') as fw:
            fw.write('User name: ' + self.user_name + '\n')
            fw.write('Begin time: ' + self.begin_time + '\n\n\n')
            fw.write('Config params: '+ '\n')
            for key, val in self.params.items():
                fw.write(str(key) + ': ' + str(val) + '\n')
            fw.write('\n\n\n')

            fw.write('Other params: ' + '\n')
            fw.write('alpha: ' + str(self.alpha) + '\n')
            fw.write('early stop num : ' + str(self.early_stop_num) + '\n')
            fw.write('num samples: ' + str(self.num_samples)+ '\n')
            fw.write('adv batch update: ' + str(self.batch_update)+ '\n')
            fw.write('pro batch update: ' + str(self.train_batch_update)+ '\n')
            fw.write('write log update: ' + str(self.write_log_update)+ '\n')
            fw.write('random init: ' + str(self.random_init)+ '\n')
            fw.write('seed: ' + str(self.seed)+ '\n')
            fw.write('\n\n\n')

            fw.write('Force config: ' + '\n')
            for item in self.adv_forces:
                fw.write(" ".join(map(str, item)))
                fw.write('\n')
            fw.write('\n')
            for key,val in self.idx_to_action.items():
                fw.write(str(key) + ':' + val)
                fw.write('\n')
            fw.write('\n')




        if self.is_human:
            with open(self.config_log_path, 'w') as fw:
                fw.write('User name: ' + self.user_name + '\n')
                fw.write('Begin time: ' + self.begin_time + '\n\n\n')
                fw.write('Config params: ' + '\n')
                for key, val in self.params.items():
                    fw.write(str(key) + ': ' + str(val) + '\n')
                fw.write('\n\n\n')

                fw.write('Other params: ' + '\n')
                fw.write('alpha: ' + str(self.alpha)+ '\n')
                fw.write('early stop num : ' + str(self.early_stop_num)+ '\n')
                fw.write('num samples: ' + str(self.num_samples)+ '\n')
                fw.write('adv batch update: ' + str(self.batch_update)+ '\n')
                fw.write('pro batch update: ' + str(self.train_batch_update)+ '\n')
                fw.write('write log update: ' + str(self.write_log_update)+ '\n')
                fw.write('random init: ' + str(self.random_init)+ '\n')
                fw.write('seed: ' + str(self.seed)+ '\n')
                fw.write('\n\n\n')

                fw.write('Force config: ' + '\n')
                for item in self.adv_forces:
                    fw.write(" ".join(map(str, item)))
                    fw.write('\n')
                fw.write('\n')
                for key,val in self.idx_to_action.items():
                    fw.write(str(key) + ':' + val)
                    fw.write('\n')
                fw.write('\n')



    def initialize_time(self, control_freq):
        self.cur_time =0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep<=0:
            raise ValueError('xml model defined non-positive time step')
        self.control_freq = control_freq
        if control_freq<=0:
            raise ValueError('control frequency is invalid')

        self.control_timestep = 1./control_freq



    def _load_model(self):
        pass



    def _get_reference(self):
        pass







    def current_fixed_center(self):
        #yyA
        center_point = [0, 0]#(-1, -1)
        if self.count_to_100 != 50:
            center_point[0] = self.min_coord + self.current_coord_offset_X
            center_point[1] = self.min_coord + self.current_coord_offset_Y
        else: # count_to_100 == 50 (0~50)
            self.count_to_100 = 0

            center_point[0] = self.min_coord + self.current_coord_offset_X
            center_point[1] = self.min_coord + self.current_coord_offset_Y
            mean_reward = sum(self.post_reward_list[1:50])/len(self.post_reward_list[1:50])

            self.post_reward_list = []


            self.R_table[center_point[0] - self.min_coord, center_point[1] - self.min_coord] = mean_reward
            print("current_center_point: {}".format(center_point))
            print("corresponding reward: {}".format(mean_reward))
            f = open('gt_center_avg_reward.txt', mode='a', encoding='utf-8')
            f.write(str(center_point))
            f.write("\n")
            f.write(str(mean_reward))
            f.write("\n")
            f.close()
            if self.current_coord_offset_X + 10 >= self.range_ - 1:
                if self.current_coord_offset_Y + 10 >= self.range_ - 1:
                    self.stop_training = True
                    f = open('gt_center_avg_reward.txt', mode='a', encoding='utf-8')
                    f.write("EOT")
                    f.close()
                    return center_point
                self.current_coord_offset_X = 0
                self.current_coord_offset_Y += 10
            else:
                self.current_coord_offset_X += 10

        self.count_to_100 += 1
        return (center_point[0], center_point[1])


    def get_ik_param(self, object_xml):

        ###added by Yoon for test
        #self.object_xml = 'bottle.xml'

        print(colored('Init scene with top-down camera', 'red'))
        path = xml_path_completion('Baxter/baxter/master1.xml')
        model = load_model_from_path(path)

        # random initialize model position/quaternion/size
        sim = MjSim(model)

        viewer = MjViewer(sim)
        viewer.cam.fixedcamid = 0
        viewer.cam.type = const.CAMERA_FIXED

        sim.step()
        # viewer.render()
        frame = viewer._read_pixels_as_in_window()
        h, w = frame.shape[:2]
        imageio.imwrite(image_path_completion('ori.jpg'), frame)
        crop_frame = frame[:, int((w - h) / 2):int((w - h) / 2) + h, :]
        crop_h, crop_w = crop_frame.shape[:2]
        self.crop_h_ = crop_h
        self.crop_w_ = crop_w
        imageio.imwrite(image_path_completion('crop.jpg'), crop_frame)





        #max point
        (center_pt, R, grasp_angle, patch_Is_resized, fc8_predictions, self.fc8_norms, R_spec, R_table_update_info) = predict_from_img(self.training_R_table_ground_truth, image_path_completion('crop.jpg'), object_xml, self.G, self.total_steps, self.log_name, is_train=self.train_pro)
        #Where to grasp when training ground truth R-table







        if self.should_load_R_table:
            self.min_coord = R_spec[1]
            if not self.is_once_created_R_table:
                self.range_ = R_spec[0]
                if self.training_R_table_ground_truth:
                    self.R_table = np.load("./R_table_gt.npy")
                else:
                    self.R_table = np.load("./R_table.npy")
                self.is_once_created_R_table = True
        else:
            self.min_coord = R_spec[1]
            if not self.is_once_created_R_table:
                self.range_ = R_spec[0]
                value = np.empty((), dtype=object)
                value[()] = (-1, -1)  # reward - angle pairs
                self.R_table = np.full([self.range_, self.range_], value, dtype=object)
                self.is_once_created_R_table = True

            # Update R_table by 128 random patches
            if not self.training_R_table_ground_truth:
                for looper in range(self.num_samples):
                    e = self.R_table[R_table_update_info[1][looper] - self.min_coord,
                                     R_table_update_info[2][looper] - self.min_coord]

                    if e[0] < (R_table_update_info[0][looper])[0]:
                        self.R_table[R_table_update_info[1][looper] - self.min_coord,
                                     R_table_update_info[2][looper] - self.min_coord] = R_table_update_info[0][looper]

        if self.training_R_table_ground_truth:
            center_pt = self.current_fixed_center()
        else:
            center_pt = predict_from_R_table(self.R_table)
            print("center from R table (Randomized order): ", center_pt)


        #for looper in range(self.num_samples):
        #    print("updated part of the R-table: ", self.R_table[R_table_update_info[1][looper] - min_coord, R_table_update_info[2][looper] - min_coord])


        #print("R_table_update_info: ", R_table_update_info[0])
        #print("R: ", R)
        print("The R- Table: ", self.R_table)
        ## test script by Yoon
        #print("here is the prediction array!: ", fc8_predictions)

        ##yy1
        print('center {}, angle {}'.format(center_pt, grasp_angle))

        quat = np.array([np.cos(grasp_angle / 2), 0, 0, -np.sin(grasp_angle / 2)])
        coeff = 0.0020
        if self.object_xml=='half-nut.xml':
            grasp_z = 0.858
        else:
            grasp_z = 0.86
        grasp_pt = (0.8 - crop_w / 2 * coeff + center_pt[0] * coeff, 0 + crop_h / 2 * coeff - center_pt[1] * coeff, grasp_z)

        print('grasping point: ', grasp_pt)

        new_quat = np.empty(4)
        functions.mju_mulQuat(new_quat, quat, np.array([0.0, 0.0, 1.0]))


        # original: provide adv action predictor input
        siz = 112
        rot_image = imutils.rotate(crop_frame, grasp_angle * 180 /np.pi)
        rot_image = rot_image[int(center_pt[1]- siz):int(center_pt[1] + siz),
                         int(center_pt[0]- siz):int(center_pt[0] + siz), :]
        # if rot_image.shape!=(224,224):
        #     padX = rot_image.shape[0]
        #     padY = rot_image.shape[1]
        #     rot_image = np.pad(rot_image, ((siz-padX//2, siz-padX-padX//2),(siz-padY//2,siz-padY-padY//2),(0,0)), 'constant')
        #     print('image being padded!')

        # shakeNet: provide adv action predictor input
        # siz = 224
        # rot_image = rotateImageAndExtractPatch(crop_frame, grasp_angle, center_pt ,siz)
        #debug: rot_image saved
        try:
            imageio.imwrite(image_path_completion('debug_shakenet_input.jpg'), rot_image)
            print('debug: debug_shakenet_input image saved at: {}'.format(image_path_completion('debug_shakenet_input')))
        except:
            print('shakenet image cannot be saved')

        if self.adv_init:      
            try:
                rot_image = preprocess(rot_image)
            except:
                print('preprocesssing for rot_image failed')
                rot_image = preprocess(crop_frame)

        # original: store rotate image
        #
        # try:
        #     imageio.imwrite(image_path_completion('rotate_crop.jpg'), crop_frame_rot)
        # except:
        #     print('error saving image: {}'.format(image_path_completion('rotate_crop.jpg')))

        if viewer is not None:
            glfw.destroy_window(viewer.window)
            viewer = None


        return (grasp_pt, new_quat, rot_image, patch_Is_resized, fc8_predictions)



    def _reset_xml(self, object_xml):
        # reset master.xml
        tree = ET.parse(xml_path_completion("Baxter/baxter/master_tmp.xml"))
        root = tree.getroot()
        root[3].attrib['file'] = object_xml
        tree.write(xml_path_completion("Baxter/baxter/master.xml"))

        # reset master1.xml
        tree = ET.parse(xml_path_completion("Baxter/baxter/master1_tmp.xml"))
        root = tree.getroot()
        root[3].attrib['file'] = object_xml
        tree.write(xml_path_completion("Baxter/baxter/master1.xml"))



    # change object, override object parameters on the fly
    def _reset_xml2(self, object_xml):

        tree = ET.parse(xml_path_completion('Baxter/baxter/{}'.format(object_xml)))
        root = tree.getroot()
        # randomize

        pos_vec = root[-1][0].attrib['pos'].strip().split(' ')
        pos_vec = [i for i in pos_vec if i!='']
        cube_pos = [float(i) for i in pos_vec]
        cube_pos_x_low = cube_pos[0] - 0.1
        cube_pos_x_high = cube_pos[0] + 0.05
        cube_pos_y_low =  cube_pos[1] - 0.1
        cube_pos_y_high = cube_pos[1] + 0.1

        #Randomizing part
        #yy0

        cube_x_new = np.random.uniform(cube_pos_x_low, cube_pos_x_high) #(cube_pos_x_high + cube_pos_x_low) / 2#
        cube_y_new = np.random.uniform(cube_pos_y_low, cube_pos_y_high) #(cube_pos_y_high + cube_pos_y_low) / 2#
        cube_x_new = (cube_pos_x_high + cube_pos_x_low) / 2#
        cube_y_new = (cube_pos_y_high + cube_pos_y_low) / 2#


        root[-1][0].attrib['pos'] = ' '.join([str(i) for i in [cube_x_new, cube_y_new, cube_pos[2]]])

        quat_vec =  root[-1][0].attrib['quat'].strip().split(' ')
        quat_vec =[i for i in quat_vec if i!='']
        body_xquat = [float(i) for i in quat_vec]

        #Randomizing part
        #radian = np.random.uniform(low= -45, high=45) * np.pi/180
        radian = 0
        if object_xml == 'bottle.xml':
            radian = 0 #45 * np.pi / 180
        elif object_xml == 'cube.xml':
            radian = -45 * np.pi / 180
        else:
            radian = 90 * np.pi / 180
        new_quat = np.empty(4)
        functions.mju_mulQuat(new_quat, np.array([np.cos(radian/2), 0., 0., np.sin(radian/2)]), np.array(body_xquat))

        if object_xml == 'bottle.xml' or object_xml == 'cube.xml':
            root[-1][0].attrib['quat'] = ' '.join([str(i) for i in new_quat])
        else:
            root[-1][0].attrib['quat'] = ' '.join([str(i) for i in np.array([np.cos(radian/2), 0., 0., np.sin(radian/2)])])


        tree.write(xml_path_completion('Baxter/baxter/tmp.xml'))

        tree = ET.parse(xml_path_completion('Baxter/baxter/master_tmp.xml'))
        root = tree.getroot()
        root[3].attrib['file'] = 'tmp.xml'
        tree.write(xml_path_completion('Baxter/baxter/master.xml'))

        tree = ET.parse(xml_path_completion('Baxter/baxter/master1_tmp.xml'))
        root = tree.getroot()
        root[3].attrib['file'] = 'tmp.xml'
        tree.write(xml_path_completion('Baxter/baxter/master1.xml'))


    def reset(self):
        print('#'*30)
        print('Round {}'.format(self.total_steps))
        print('#'*30)
        gc.collect()
        if not self.random_init:
            self.object_xml = np.random.choice([self.object_xml])
        else:
            self.object_xml = np.random.choice(['bottle.xml', 'cube.xml' ,'round-nut.xml', 'half-nut.xml', 'new_cube.xml'])

        #Added by Yoon for test

        opt = self.option #np.random.randint(0, 5) #0 5

        if opt == 0:
            self.object_xml = 'bottle.xml'
        elif opt == 1:
            self.object_xml = 'new_cube.xml'
        elif opt == 2:
            self.object_xml = 'cube.xml'
        elif opt == 3:
            self.object_xml = 'half-nut.xml'
        elif opt == 4:
            self.object_xml = 'round-nut.xml'

        ######self.object_xml = 'bottle.xml' #150
        ######self.object_xml = 'new_cube.xml' #150
        ###self.object_xml = 'round-nut.xml' #150
        ##self.object_xml = 'cube.xml' #200
        ##self.object_xml = 'half-nut.xml' #150


        print(colored('current object: {}'.format(self.object_xml), 'red'))
        if self.random_init:
            self._reset_xml2(self.object_xml)
        else:
            self._reset_xml(self.object_xml)
        self._reset_internal()
        self.sim.forward()

        # debug, get param for ik
        #os.system("free -m")
        (grasp_pt, new_quat, rot_image, self.patch_Is_resized, self.fc8_predictions) = self.get_ik_param(self.object_xml)
        print('debug reset fc8_predictions shape: ', self.fc8_predictions.shape)

        di = self._get_observation()
        di['grasp_pt'] = grasp_pt
        di['new_quat'] = new_quat

        # original: prepare adv_input
        # X_batch = prepare_X_batch(rot_image)
        # di['X_batch'] = X_batch
        # self.X_batch = X_batch


        #original: perform adv_action prediction
        # if not self.is_human and self.adv_init:
        #     y_prob = self.adv_policy['sess'].run(self.adv_policy['y_prob'], feed_dict={self.adv_policy['X']:X_batch})
        #     print(colored('adv y_prob: {}'.format(y_prob),'red'))
        #     self.adv_prob = np.max(y_prob[0])
        #     print(colored('max adv_prob: {}'.format(self.adv_prob), 'blue'))
        #     adv_action = np.argmax(y_prob[0])
        #     print(colored('adv_action prediction: {}/{}'.format(adv_action, self.n_adv_outputs),'blue'))
        #     di['adv_action'] = adv_action


        # shakeNet: prepare adv_input in the function
        if self.adv_init:
            self.X_batch = prepare_X_batch2(rot_image)
        #debug, use shakeNet 
        if not self.is_human and self.adv_init:
            adv_probs, adv_action = adv_predict_from_img(rot_image, self.G_adv)
            print(colored('adv_action prediction: {}/{}/{}'.format(adv_action, self.n_adv_outputs, self.idx_to_action[adv_action]),'blue'))
            di['adv_action'] = adv_action
            self.adv_prob = np.max(adv_probs[0])
            print(colored('adv_prob_vec: {}, adv_max_prob: {}'.format(adv_probs, self.adv_prob),'cyan'))


        return di



    def _reset_internal(self):
        self._load_model()
        self.mjpy_model = self.model.get_model(mode='mujoco_py')
        self.sim= MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)


        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = False



        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time =0
        self.timestep=0
        self.done = False

        # set speed and pert id
        # self.viewer.viewer._run_speed = self.speed
        # self.viewer.viewer.pert.select = self.pert

        #debug



    def _get_observation(self):
        return OrderedDict()

    def average_val_R_table_deck(self, deck):

        sum_reward = 0
        sum_angle = 0

        for e in deck:
            sum_reward += np.exp(e[0])
            sum_angle += e[1]
        return (sum_reward/len(deck), sum_angle/len(deck))

    def Approximate_R_value(self, X, Y):
        #yyB

        X = int(X)
        Y = int(Y)

        minnum = int(self.min_coord)
        range_ = int(self.range_)
        maxnum = minnum + range_ - 1
        N = 1
        deck = []
        approximate_R_val = None

        x = None
        y = None

        min_phase_num = 20
        max_phase_num = 40

        while (len(deck) == 0 or min_phase_num >= 0) and max_phase_num != 0:
            y = Y + (N-1)
            if y>= maxnum:
                y = maxnum
            for ind in range(X - (N - 1), X + (N - 1) + 1):
                x = ind
                if x<=minnum:
                    x = minnum
                elif x>=maxnum:
                    x = maxnum
                if self.R_table[x - minnum, y - minnum] != (-1, -1):
                    deck.append(self.R_table[x - minnum, y - minnum])
            x = X - (N-1)
            if x<=minnum:
                x = minnum
            for ind in range(Y - (N - 1), Y + (N - 1) + 1):
                y = ind
                if y<=minnum:
                    y = minnum
                elif y>=maxnum:
                    y = maxnum
                if self.R_table[x - minnum, y - minnum] != (-1, -1):
                    deck.append(self.R_table[x - minnum, y - minnum])
            y = Y - (N-1)
            if y<=minnum:
                y = minnum
            for ind in range(X - (N - 1), X + (N-1) + 1):
                x = ind
                x = ind
                if x<=minnum:
                    x = minnum
                elif x>=maxnum:
                    x = maxnum
                if self.R_table[x - minnum, y - minnum] != (-1, -1):
                    deck.append(self.R_table[x - minnum, y - minnum])
            x = X + (N-1)
            if x>=maxnum:
                x = maxnum
            for ind in range(Y - (N - 1), Y + (N - 1) + 1):
                y = ind
                if y<=minnum:
                    y = minnum
                elif y>=maxnum:
                    y = maxnum
                if self.R_table[x - minnum, y - minnum] != (-1, -1):
                    deck.append(self.R_table[x - minnum, y - minnum])

            N += 1
            min_phase_num -= 1
            max_phase_num -= 1

        if deck != []:
            self.R_table[X - minnum, Y- minnum] = self.average_val_R_table_deck(deck)
        else:
            self.R_table[X - minnum, Y - minnum] = (-1, -1)

        return self.R_table[X - minnum, Y- minnum]


    #yy4
    def categorize_force(self, target_pos2, net_displacement, post_reward):


        if post_reward == 0.8:
            return -1 #effective adv
        #yyt
        crop_h = self.crop_h_
        crop_w = self.crop_w_
        coeff = 0.0020

        #print("target_pos2: ", target_pos2)

        pixel_x1 = (target_pos2[0] - 0.8) / coeff + crop_w / 2
        pixel_y1 = -(target_pos2[1]) / coeff + crop_h / 2

        pixel_x2 = (net_displacement[0] + target_pos2[0] - 0.8) / coeff + crop_w / 2
        pixel_y2 = -(net_displacement[1] + target_pos2[1]) / coeff + crop_h / 2

        print("pixel_x1: ", pixel_x1)
        print("pixel_y1: ", pixel_y1)
        print("pixel_x2: ", pixel_x2)
        print("pixel_y2: ", pixel_y2)

        if pixel_x2 >= self.min_coord + self.range_:
            pixel_x2 = self.min_coord + self.range_ - 1
        if pixel_y2 >= self.min_coord + self.range_:
            pixel_y2 = self.min_coord + self.range_ - 1
        if pixel_x2 <= self.min_coord:
            pixel_x2 = self.min_coord
        if pixel_y2 >= self.min_coord:
            pixel_y2 = self.min_coord

        #pixel_x1 = pixel_x1 + self.min_coord
        #pixel_y1 = pixel_y1 + self.min_coord

        #pixel_x2 = pixel_x2 + self.min_coord
        #pixel_y2 = pixel_y2 + self.min_coord

        R_val1 = self.Approximate_R_value(pixel_x1, pixel_y1)[0]
        R_val2 = self.Approximate_R_value(pixel_x2, pixel_y2)[0]

        classifier = R_val2 - R_val1 #self.R_table[pixel_x2, pixel_y2][0] - self.R_table[pixel_x1, pixel_y1][0]
        print("derivative of the reliability: ", classifier)
        if classifier >= 0 :
            return 1 #collaborative
        elif classifier < 0:
            return 0 #ineffective adv


    def pro_adv_apply_action(self, pro_action, adv_action=None):

        ###added by Yoon for training
        np.random.seed(int(time.time()))

        target_pos2 = pro_action[0]
        target_quat = pro_action[1].astype('float64')
        target_pos1 = np.append(target_pos2[:2],[1.4])
        model_name = xml_path_completion('Baxter/baxter/master.xml')
        max_step = 5000

        quat_z = np.array([math.cos(math.pi / 4.), 0.0, 0.0, math.sin(math.pi / 4.)])
        new_quat = np.empty(4)
        functions.mju_mulQuat(new_quat, quat_z, target_quat)

        idx = []
        idx.extend(self._ref_joint_pos_indexes)
        idx.extend(self._ref_gripper_joint_pos_indexes)

        num = 6
        qpos = []
        for n in range(0, num+1):
            cpos = target_pos1 + (target_pos2 - target_pos1) / num *n
            qpos.append(inverse_kinematics(model_name ,body_name = "left_gripper_base",
                                           target_pos = cpos, target_quat = new_quat, image_name = "a"))


        model = load_model_from_path(model_name)
        sim = MjSim(model)
        # random initialize model as the same as master1.xml
        viewer = MjViewer(sim)

        

        viewer._run_speed = self.speed
        viewer.pert.select = self.pert


        #'l_gripper_l_finger', 'l_gripper_l_finger_tip', 'l_gripper_r_finger', 'l_gripper_r_finger_tip'
        L_finger_p1_idx = sim.model.body_name2id('l_gripper_l_finger')
        L_finger_p2_idx = sim.model.body_name2id('l_gripper_l_finger_tip')
        R_finger_p1_idx =sim.model.body_name2id('l_gripper_r_finger')
        R_finger_p2_idx = sim.model.body_name2id('l_gripper_r_finger_tip')

        Gripper_idx = [L_finger_p1_idx, L_finger_p2_idx, R_finger_p1_idx, R_finger_p2_idx]


        step = 1
        print('move to qpos1')
        sim.data.ctrl[:] = qpos[3][idx]
        sim.data.ctrl[16] = 0.2
        sim.data.ctrl[17] = -0.2


        while step <= max_step / 2:
            ind = sim.model.body_name2id('object0')
            sim.data.xfrc_applied[ind][2] = -0.02
            sim.step()
            if self.use_render:
                viewer.render()
            step += 1

        print('slowly move to qpos2')
        for n in range(1, num+1):
            sim.data.ctrl[:] = qpos[n][idx]
            sim.data.ctrl[16] = 0.2
            sim.data.ctrl[17] = -0.2
            step =1
            while step <= max_step / num:
                ind = sim.model.body_name2id('object0')
                # sim.data.xfrc_applied[ind][2] = -0.1
                sim.step()
                if self.use_render:
                    viewer.render()
                step += 1

        Obj_id = sim.model.body_name2id('object0')

        """"""
        window = viewer.window
        window_size = glfw.get_window_size(window)
        print("window_size: ", window_size)

        """
        width = window_size[0]
        height = window_size[1]

        tableX = sim.data.body_xpos[self.table_body_id][0]
        # tableY = sim.data.body_xpos[self.table_body_id][1]

        applying_force = np.array([0, 0, 0, 0, 0, 0])
        scaleFactor = 1.8

        X_clicked = 10
        Y_clicked = 10

        force_scale_factor = 0.5

        ever_mouse_pressed = False

        xf = 0
        yf = 0
        isX_set = False
        isY_set = False


        init_cam = []
        init_cam.append(copy.deepcopy(viewer.cam.lookat))
        init_cam.append(viewer.cam.distance)
        init_cam.append(viewer.cam.elevation)
        init_cam.append(viewer.cam.azimuth)

        
        while True:
            if viewer.key_pressed == glfw.KEY_Z:
                break

            viewer.cam.lookat[0] = 0.8
            viewer.cam.lookat[1] = 0
            viewer.cam.lookat[2] = 0
            viewer.cam.distance = model.stat.extent * 1.0
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 90

            viewer.render()
            #sim.step()
            #viewer.cam = modified_cal
            if viewer._button_left_pressed:
                X_clicked = tableX + (scaleFactor*width/height)*(viewer._last_mouse_x - width/2)/width
                Y_clicked = scaleFactor*(-viewer._last_mouse_y + height/2)/height

                if X_clicked > 1.6:
                    X_clicked = 1.6
                elif X_clicked < 0:
                    X_clicked = 0
                if Y_clicked > 0.85:
                    Y_clicked = 0.85
                elif Y_clicked < -0.85:
                    Y_clicked = -0.85
                #print("x: ", X_clicked)
                #print("y: ", Y_clicked)
                ever_mouse_pressed = True
            if ever_mouse_pressed:
                xf = X_clicked - sim.data.body_xpos[Obj_id][0]
                yf = Y_clicked - sim.data.body_xpos[Obj_id][1]
                xdist = np.abs(sim.data.body_xpos[Obj_id][0] - X_clicked)
                ydist = np.abs(sim.data.body_xpos[Obj_id][1] - Y_clicked)
                xf = xf*force_scale_factor*np.sqrt(xdist*xdist*xdist)
                yf = yf*force_scale_factor*np.sqrt(ydist*ydist*ydist)
                # viewer.render()
                # if viewer._button_left_pressed:
                #print("obj pos:", sim.data.body_xpos[Obj_id])
                if np.abs(sim.data.body_xpos[Obj_id][0] - X_clicked) < 0.0002:
                    xf = 0
                    print("X coord achieved!")
                    isX_set = True
                if np.abs(sim.data.body_xpos[Obj_id][1] - Y_clicked) < 0.0002:
                    yf = 0
                    print("Y coord achieved!")
                    isY_set = True
                #if isX_set and isY_set:
                #    break

                if xf > 0.1:
                    xf = 0.1
                if yf > 0.1:
                    yf = 0.1

                applying_force = np.array([xf, yf, 0, 0, 0, 0])
                sim.data.xfrc_applied[Obj_id][:] = applying_force
                sim.step()
            #sleep(0.01)

        viewer.cam.lookat[:] = init_cam[0][:]
        viewer.cam.distance = init_cam[1]
        viewer.cam.elevation = init_cam[2]
        viewer.cam.azimuth = init_cam[3]
        viewer.render()
        sim.step()
        """

        viewer.key_pressed = None


        #added by Yoon
        #aux_reward = 0
        #init_check = None
        #ever_touched = False

        ##yy2_0
        objX = sim.data.body_xpos[Obj_id][0]
        objY = sim.data.body_xpos[Obj_id][1]
        #print("Object Coords on Table: {}, {}".format(objX, objY))
        #print("Object Quat on Table: {}".format(sim.data.body_xquat[Obj_id]))
        quat_lifted = sim.data.body_xquat[Obj_id]
        #print("quat_on_table: ", quat_lifted)

        approx_angle_0 = 2*np.arccos(quat_lifted[0])
        #print("radian angle: ", approx_angle_0)

        #print('grasp object')
        sim.data.ctrl[16] = 0
        sim.data.ctrl[17] = 0
        step =1

        while step <= max_step / num:

            # added by Yoon
            #if not ever_touched:
            #    if self.object_xml == 'round-nut.xml' or self.object_xml == 'half-nut.xml' or self.object_xml == 'new_cube.xml':
            #        init_check = self._adv_check3(sim)
            #    else:
            #        init_check = self._adv_check_middle(sim=sim)
            #    if not init_check:
            #        aux_reward = 0.08
            #        ever_touched = True

            ind = sim.model.body_name2id('object0')
            # sim.data.xfrc_applied[ind][2] = -0.1
            sim.step()
            if self.use_render:
                viewer.render()
            step +=1

        #print("Ever touched: ", ever_touched)


        adv_action = 0
        recordState_alter = []
        sub_vector = []
        ed = -1

        viewer.key_pressed = None


        print('lift object')  
        for n in range(num-1, -1, -1):
            sim.data.ctrl[:] = qpos[n][idx]
            sim.data.ctrl[16] = -0.2
            sim.data.ctrl[17] = 0.2
            step = 1
            while step <= max_step/8:
                sim.step()
                if self.use_render:
                    viewer.render()
                step += 1

        # record grasping point
        #viewer.vopt.geomgroup[0]=0
        #viewer.vopt.geomgroup[1]=0
        
        # # doing nothing, original /30, now to increase slipping time
        step=1
        div = 30
        if not self.train_pro:
            div = 1.
        while step<=max_step/div:
            ind = sim.model.body_name2id('object0')
            sim.step()
            if self.use_render:
                viewer.render()
            step += 1

        gravity=np.zeros(6)
        gravity[2] = -0.1


        #increase time of doing nothing
        if self.object_xml =='round-nut.xml' or self.object_xml=='half-nut.xml' or self.object_xml=='new_cube.xml':
            nut_id = sim.model.body_name2id('object0')
            self.ori_nut_pos = copy.deepcopy(sim.data.body_xpos[nut_id])
            sleep(0.1)



        adv_error = 0.0
        pre_reward = 0.0
        lift_success = self._check_success2(sim)



        save_path_for_force_filter2 = None
        if self.object_xml == "bottle.xml":
            save_path_for_force_filter2 = './data/bottle_2_3.txt'
        elif self.object_xml == "new_cube.xml":
            save_path_for_force_filter2 = './data/new_cube_2_3.txt'
        elif self.object_xml == "cube.xml":
            save_path_for_force_filter2 = './data/cube_2_3.txt'
        elif self.object_xml == "half-nut.xml":
            save_path_for_force_filter2 = './data/half-nut_2_3.txt'
        elif self.object_xml == "round-nut.xml":
            save_path_for_force_filter2 = './data/round-nut_2_3.txt'


        Option = None
        adv_pred = np.ones(1).reshape(1, 1)
        if self.object_xml == 'bottle.xml':
            Option = 0
        elif self.object_xml == 'new_cube.xml':
            Option = 1
        elif self.object_xml == 'cube.xml':
            Option = 2
        elif self.object_xml == 'half-nut.xml':
            Option = 3
        elif self.object_xml == 'round-nut.xml':
            Option = 4

        isKeyEverPressed = False



        if not lift_success:
            if not self.use_force_filter:
                if self.is_test:
                    with open('./data/f_contained_curriculum_test.txt', 'a') as f:
                        f.write(str(-1) + "\n")
                else:
                    with open('./data/f_contained_curriculum.txt', 'a') as f:
                        f.write(str(-1) + "\n")

        # if height of cube satisfies, perform perturbation

        #print("total setep : ", self.total_steps)
        #print("apply intention: ", self.intention)

        self.is_valid_sample = True
        self.lift_success = False
        self.lift_success = lift_success

        Loc1 = np.zeros(2)#(0, 0)
        Loc2 = np.zeros(2)#(0, 0)
        approx_angle_1 = 0
        #_adv_check_middle
        if lift_success:
            print(colored('first check success','green'))
            if self.to_train:
                self.rot_images.append(self.X_batch)
                self.train_nums+=1
            pre_reward +=1
            ind = sim.model.body_name2id('object0')
            # save(sim,viewer)

            if self.is_human or self.adv_init or self.random_perturb:
                print(colored('perform perturbation now!','blue'))
                print(colored('perturbation: {}'.format(self.idx_to_action[adv_action]), 'yellow'))
                step =1
                human_step =1

                ##yy2_1
                objX = sim.data.body_xpos[Obj_id][0]
                objY = sim.data.body_xpos[Obj_id][1]
                print("Object Coords before applying a force: {}, {}".format(objX, objY))
                mean_gripper_pos = sum(sim.data.body_xpos[Gripper_idx]) / 4  # 4 col
                Loc1[0] = mean_gripper_pos[0] - objX
                Loc1[1] = mean_gripper_pos[1] - objY
                print("Loc1: ", Loc1)

                quat_lifted = sim.data.body_xquat[Obj_id]
                print("quat_lifted: ", quat_lifted)
                approx_angle_1 = 2 * np.arccos(quat_lifted[0])#*180/np.pi
                print("radian angle: ", approx_angle_1)

                while True:
                    if self.is_human and viewer.key_pressed==glfw.KEY_Q:
                        break
                    if not self.is_human and step == max_step/25:
                        break
                    if self.is_human:
                        if not isKeyEverPressed:

                            if self.is_data_collect_mode:
                                np.random.seed(int(time.time()))
                            np.random.seed(int(time.time()))


                            adv_action = np.random.randint(1, 5) #self.human_perturb_action2(viewer) #
                            #adv_action = self.human_perturb_action2(viewer) #
                            self.force_random_modifier()


                            #self.use_force_filter = True
                            if adv_action != 0: #and self.use_force_filter:
                                ## test code - added by Yoon 05.14
                                #print("gripper quat: ", sim.data.body_xquat[Gripper_idx][0])
                                gripper_quat = sim.data.body_xquat[Gripper_idx][0]  # 4 col
                                #print("gripper pos: ", (sim.data.body_xpos[Gripper_idx]))
                                mean_gripper_pos = sum(sim.data.body_xpos[Gripper_idx]) / 4  # 4 col
                                gripper_0 = sim.data.body_xpos[Gripper_idx][0]
                                gripper_2 = sim.data.body_xpos[Gripper_idx][2]
                                #print("averaged gripper pos: ", )
                                """
                                itr_t = 0
                                for q in np.nditer(sim.data.body_xquat[Obj_id]):
                                    # recordState.append(float(np.asscalar(q)))
                                    recordState_alter.append(float(np.asscalar(q)))
                                    itr_t += 1

                                sub_vector.append(copy.deepcopy(mean_gripper_pos))
                                #sim.data.body_xmat[Gripper_idx][0]
                                sub_vector.append(copy.deepcopy(sim.data.body_xpos[Obj_id]))

                                sub_vector.append(copy.deepcopy(gripper_0))
                                sub_vector.append(copy.deepcopy(gripper_2))

                                for v in np.nditer(gripper_quat):  # sim.data.ctrl[:]:
                                    recordState_alter.append(float(np.asscalar(v)))

                                iter_idx = 0
                                for v in np.nditer(mean_gripper_pos):  # sim.data.ctrl[:]:
                                    temp = np.asscalar(v) - np.asscalar(sim.data.body_xpos[Obj_id][iter_idx])
                                    recordState_alter.append(temp)
                                ## test code - added by Yoon 05.14
                                recordState_alter.append(adv_action)
                                """

                                """
                                if not self.is_data_collect_mode:
                                    recordState_alter.append(0)
                                    if self.is_test:
                                        total_dim = len(recordState_alter)
                                        if not self.is_once_created_filter:
                                            self.G_force = init_force_filter2(is_alt=True, opt=Option, batch_size=1,
                                                                     gpu_id=0,
                                                                     lr_rate=1e-4,
                                                                     Collective_dimension=total_dim - 1)
                                            self.is_once_created_filter = True

                                        vicious_actions = []
                                        for actions in range(1, 7):
                                            recordState_alter[11] = actions
                                            feed_column = np.array(recordState_alter, dtype=float).reshape(1, total_dim,
                                                                                                           1,
                                                                                                           1)
                                            adv_pred = self.G_force._session_controle(feed_column, is_train=False,
                                                                                 should_reuse=False)  # 0 ~1
                                            if 0.5 > adv_pred:
                                                vicious_actions.append(actions)

                                        actions_length = len(vicious_actions)
                                        if actions_length == 0:
                                            adv_action = np.random.randint(1, 5)#6
                                        else:
                                            adv_action = vicious_actions[np.random.randint(0, actions_length)]
                                            if adv_action == 6:
                                                adv_action = vicious_actions[np.random.randint(0, actions_length)]
                                        print("vicious actions set: ", vicious_actions)

                                    else:
                                        total_dim = len(recordState_alter)
                                        feed_column = np.array(recordState_alter, dtype=float).reshape(1, total_dim,
                                                                                                       1,
                                                                                                       1)
                                        if not self.is_once_created_filter:
                                            self.G_force = init_force_filter2(is_alt=True, opt=Option, batch_size=1,
                                                                     gpu_id=0,
                                                                     lr_rate=1e-4,
                                                                     Collective_dimension=total_dim - 1)
                                            self.is_once_created_filter = True

                                        adv_pred = self.G_force._session_controle(feed_column, is_train=False,
                                                                             should_reuse=False)  # 0 ~1
                                """


                                isKeyEverPressed = True

                                """
                                ed = 0
                                ed_alt1 = 0
                                ed_alt2 = 0
                                ed_alt3 = 0
                                for i in range(2):
                                    temp_ = np.abs(sub_vector[0][i] - sub_vector[1][i])
                                    temp2 = np.abs(sub_vector[2][i] - sub_vector[1][i])
                                    temp3 = np.abs(sub_vector[3][i] - sub_vector[1][i])
                                    temp4 = np.abs(sub_vector[2][i] - sub_vector[3][i])

                                    ed += temp_ * temp_  # np.abs(sub_vector_quat[0][i] - sub_vector_quat[1][i])
                                    ed_alt1 += temp2 * temp2
                                    ed_alt2 += temp3 * temp3
                                    ed_alt3 += temp4 * temp4

                                ed = np.sqrt(ed)
                                ed_alt1 = np.sqrt(ed_alt1)
                                ed_alt2 = np.sqrt(ed_alt2)
                                ed_sub = np.abs(ed_alt1 - ed_alt2)
                                ed_sub2 = np.sqrt(ed_alt3)

                                if self.is_data_collect_mode:
                                    self.modify_force(ed, ed_sub, ed_sub2, viewer, sim, ind)
                                """
                        if adv_action != 0:
                            isKeyEverPressed = True
                            human_step += 1
                        if human_step == 2*max_step / 25:


                            if self.use_force_filter:
                                if adv_pred[0, 0] == 0:
                                    self.is_valid_sample = True
                                    print("adv!")
                                else:
                                    self.is_valid_sample = False
                                    print("help!")
                            else:
                                self.is_valid_sample = True

                            print("adv_action took!: ", adv_action)

                            break

                    sim.data.xfrc_applied[ind][:] = self.adv_forces[int(adv_action)]
                    sim.step()

                    if self.use_render:
                        viewer.render()
                    step += 1



            # increase time of being second check
            count_step=0
            if self.object_xml=='round-nut.xml' or self.object_xml=='half-nut.xml':
                max_step*=3
            while count_step<max_step/3:
                sim.step()
                if self.use_render:
                    viewer.render()
                count_step+=1
            # adv_error =1 if height of cube is above threshold, adv_perturb fails
            #if self._check_success3(sim):
            #_adv_check2 is True is not grasp
            if self.object_xml =='round-nut.xml' or self.object_xml =='half-nut.xml' or self.object_xml=='new_cube.xml':
                second_check = self._adv_check3(sim)
            else:
                second_check = self._adv_check2(sim)

            if second_check==False:
                adv_error = 1.0
                # self.rot_y_batches.append(np.zeros((1, self.n_adv_outputs)))
                self.rot_y_batches.append(adv_error)

            #  adv_error =0, perturb success, no error will be backpro
            else:
                print(colored('second check fail', 'green'))
                adv_error = 0.0
                tmp = np.zeros((1, self.n_adv_outputs))
                self.rot_y_batches.append(adv_error)
            self.adv_error_logs.append((self.total_steps, adv_error, adv_action, self.idx_to_action[adv_action]))


        # yy2_2
        objX = sim.data.body_xpos[Obj_id][0]
        objY = sim.data.body_xpos[Obj_id][1]
        print("Object Coords after applying a force: {}, {}".format(objX, objY))
        mean_gripper_pos = sum(sim.data.body_xpos[Gripper_idx]) / 4  # 4 col
        Loc2[0] = mean_gripper_pos[0] - objX
        Loc2[1] = mean_gripper_pos[1] - objY
        print("Loc2: ", Loc2)

        # Loc_sub = (Loc1[0] - Loc2[0], Loc1[1] - Loc2[1], 0.)
        quat_lifted = sim.data.body_xquat[Obj_id]
        # print("quat_lifted_applied: ", quat_lifted)
        approx_angle_2 = 2 * np.arccos(quat_lifted[0])
        # print("radian angle: ", approx_angle_2)

        delta_angle2 = approx_angle_2 - approx_angle_1 #yyy
        # mju_rotVecQuat

        new_vec1 = np.empty(3)
        functions.mju_rotVecQuat(new_vec1, Loc2,
                                 np.array([np.cos(-delta_angle2), 0., 0., np.sin(-delta_angle2 / 2)]))
        Loc_sub = np.zeros(3)
        Loc_sub[0] = Loc1[0] - new_vec1[0]
        Loc_sub[1] = Loc1[1] - new_vec1[1]
        Loc_sub[2] = 0.

        new_vec2 = np.empty(3)
        functions.mju_rotVecQuat(new_vec2, Loc_sub,
                                 np.array([np.cos((-approx_angle_1 + approx_angle_0) / 2), 0., 0.,
                                           np.sin((-approx_angle_1 + approx_angle_0) / 2)]))

        ###debug
        if lift_success==False:
            post_reward =pre_reward
        else:
            if adv_error == 0:
                if self.is_human: 
                    self.adv_prob = 1
                elif self.random_perturb:
                    self.adv_prob = 1
                elif not self.adv_init:
                    self.adv_prob = 0
                post_reward = pre_reward-self.alpha*self.adv_prob
            else:
                post_reward = pre_reward
        print(colored('pre_reward: {}, post_reward: {}, adv_error: {}'.format(pre_reward, post_reward, adv_error), 'blue'))
        self.reward_logs.append((self.total_steps, pre_reward, post_reward))

        # close viewer of self first, then destroy this viewer
        self._destroy_viewer()
        if viewer is not None:
            glfw.destroy_window(viewer.window)
            viewer = None


        #Added by Yoon
        print("is using filter? : ", self.use_force_filter)
        if len(sub_vector) != 0 and self.is_data_collect_mode:

            self.intention = self.expert_intention(adv_action, post_reward)
            recordState_alter.append(self.intention)
            # recordState_alter.append(0)
            #print("record state: ", recordState_alter)

            # if Option != 2:
            with open(save_path_for_force_filter2, 'a') as f:
                f.write(str(recordState_alter) + "\n")

        #return (post_reward + aux_reward), adv_error
        #yy3
        if not self.training_R_table_ground_truth:
            cat = self.categorize_force(target_pos2, new_vec2, post_reward)
            print("category of the applied force: ", cat)
        if self.training_R_table_ground_truth:
            if post_reward > 0 and post_reward < 1:
                self.post_reward_list.append(0.5)
            else:
                self.post_reward_list.append(post_reward)


        return (post_reward), adv_error



    def force_random_modifier(self):
        force = 3.5
        coeff = np.random.randint(35)/10
        force *= coeff
        self.adv_forces = np.array(
            [[0, 0, 0, 0, 0, 0], [-force, 0, 0, 0, 0, 0], [0, -force, 0, 0, 0, 0], [force, 0, 0, 0, 0, 0],
                 [0, force, 0, 0, 0, 0], [0, 0, force, 0, 0, 0], [0, 0, -force, 0, 0, 0]])
        return




    def modify_force(self, ed, ed_sub, ed_sub2, viewer, sim, ind):

        #np.random.seed(int(time.time()))
        force = 3.5
        down_force = 0.3*20

        upforce = 3.5

        if self.object_xml == 'bottle.xml':
            if (ed < 0.045 and ed > 0.02) and ed_sub2 > 0.065:
                force *= 3.5
                print("portion 0")
            elif ed_sub2 > 0.065 and (ed >= 0.016 and ed <= 0.02):
                force *= 2.7  # 2.7
                print("portion 1-1")

            elif ed_sub2 > 0.065 and (ed >= 0.045 and ed <= 0.07):
                force *= 1.8  # 2.7
                print("portion 1-2")

            elif (ed >= 0 and ed < 0.016) and ed_sub2 > 0.065:
                force *= 2.52
                print("portion 2-1")
            elif (ed >= 0 and ed < 0.016) and ed_sub2 <= 0.065:
                force *= 2.42
                print("portion 2-2")
            elif ed_sub2 > 0.049:
                force *= 1.7
                print("portion 3")
            elif ed_sub2 > 0.042:
                force *= 1.2
                print("portion 4")
            else:
                force *= 1.17 #1.6
                print("portion 5")
        elif self.object_xml == 'cube.xml':
            if ed < 0.1:
                for i in range(350):
                    sim.data.xfrc_applied[ind][:] = self.adv_forces[5]
                    sim.step()
                    # if step % 1000 == 0:
                    #    print(colored('perturbation applied: {}'.format(self.idx_to_action[adv_action]), 'yellow'))
                    if self.use_render:
                        viewer.render()

            if ed <= 0.015:
                force *= 1.85#1.7#2.5
            elif ed <= 0.04 and ed > 0.015:
                force *= 1.55#1.7
            elif ed <= 0.06 and ed > 0.04:
                force *= 1.4#1.5
            elif ed <= 0.075 and ed > 0.06:
                force *= 1.25#1.35
            elif ed <= 0.095 and ed > 0.075:
                force *= 1.2#1.3
            else:
                force *= 1.1
        elif self.object_xml == 'new_cube.xml':
            if ed <= 0.06:
                force *= 1.8
            else:
                force *= 1.3
        elif self.object_xml == 'half-nut.xml':

            for i in range(350):
                sim.data.xfrc_applied[ind][:] = self.adv_forces[5]
                sim.step()
                # if step % 1000 == 0:
                #    print(colored('perturbation applied: {}'.format(self.idx_to_action[adv_action]), 'yellow'))
                if self.use_render:
                    viewer.render()

            if ed_sub < 0.06 and ed_sub > 0.02:
                if ed > 0.135:
                    force *= 1.5 #type abnormal1
                else:
                    force *= 6 #type optimal
            elif ed_sub > 0.07 and ed <= 0.08:
                force *= 1.2 #type 3
            elif ed_sub > 0.14:
                force *= 1.2
                # type abnormal2
            elif ed_sub < 0.01:
                if ed < 0.12:
                    force *= 1.65
                else:
                    force *= 1.4

        else:
            print("ed: ", ed)
            if ed > 0.1:
                force *= 1.2
            else:
                force *= 4

        #print("force: ", force)
        print("ed: ", ed)
        #print("ed-sub: ", ed_sub)
        print("ed-sub2: ", ed_sub2)
        self.adv_forces = np.array(
            [[0, 0, 0, 0, 0, 0], [-force, 0, 0, 0, 0, 0], [0, -force, 0, 0, 0, 0], [force, 0, 0, 0, 0, 0],
                 [0, force, 0, 0, 0, 0], [0, 0, upforce, 0, 0, 0], [0, 0, -down_force, 0, 0, 0]])







    def coord_modification(self, loc_candidates_angle_list):
        return loc_candidates_angle_list



    def expert_intention(self, adv_action, post_reward):

        print("post reward: ", post_reward)

        if adv_action == 6:
            return 0

        if post_reward == 1:
            return 1
        elif post_reward < 1:
            return 0


    # 0:nothing; 1:left; 2:outward; 3: down; 4:right; 5:inside; 6:up
    def human_perturb_action(self, viewer):
        adv_action = 0
        c = viewer.theta_discrete * 180/np.pi
        if( c> 0 and c< 30) or (c>330 and c<360):
            adv_action = 4
        elif (c>30 and c<60) or (c>120 and c<150):
            adv_action = 5
        elif (c>60 and c<120):
            adv_action= 6
        elif (c>150 and c< 210):
            adv_action = 1
        elif (c>210 and c<240) or (c>300 and c<330):
            adv_action = 2
        elif (c>240 and c<300):
            adv_action=3

        return adv_action

    # self.idx_to_action = {0: 'nothing', 1: 'left', 2: 'outward',  3: 'right', 4: 'inside', 5: 'up'}
    def human_perturb_action2(self, viewer):
        adv_action =0
        c = viewer.key_pressed
        if c==glfw.KEY_UP:
            adv_action = 5
        elif c == glfw.KEY_DOWN:
            adv_action = 6
        elif c == glfw.KEY_LEFT:
            adv_action = 1
        elif c== glfw.KEY_P: #RIGHIT:
            adv_action = 3
        elif c== glfw.KEY_U:
            adv_action = 4
        elif c==glfw.KEY_O:
            adv_action = 2

        elif c==glfw.KEY_Z:
            adv_action = 7
        elif c==glfw.KEY_X:
            adv_action = 8
        elif c==glfw.KEY_C:
            adv_action = 9
        elif c==glfw.KEY_B:
            adv_action = 10
        return adv_action



    # not used for now, but contains reward shaping
    def reward2(self, sim):

        reaching_reward =0
        # reward shaping
        if self.reward_shaping:
            #reaching reward
            cube_body_id = sim.model.body_name2id('object0')
            cube_pos = sim.data.body_xpos[cube_body_id]
            eef_site_id = sim.model.site_name2id('grip')
            gripper_site_pos = sim.data.site_xpos[eef_site_id]
            dist = np.linalg.norm(gripper_site_pos[:2] - cube_pos[:2])
            reaching_reward = 1 - np.tanh(10.0 *dist)
            print(colored('reaching reward: {}'.format(reaching_reward),'green'))
        return reaching_reward



    '''
    param:
    input:
    1. grasp_pt, new_quat for protagonist action
    2. adv_action predicted by adversary
    
    implement:
    1. protagonist grasps object
    2. when object is lifted, apply adversary action
    
    output:
    1. reward obtained by protagonist
    2. reward obtained by adversary
    3. return obs, reward, done, info
    '''
    def step(self, action):

        ###added by Yoon for test
        #self.object_xml = 'bottle.xml'


        # protagonist input param
        grasp_pt = action['grasp_pt']
        new_quat = action['new_quat']
        pro_input = (grasp_pt, new_quat)

        #print("grasp pt shape: ", grasp_pt.shape)
        #print("new quat shape: ", new_quat.shape)
        print("grasp pt: ", grasp_pt)
        print("new quat: ", new_quat)

        print('perform grasping and perturbation')
        if not self.is_human and self.adv_init:
            adv_input = action['adv_action']
        else:
            if self.is_human:
                adv_input =0
            elif self.random_perturb:
                adv_input = np.random.randint(0, self.adv_forces.shape[0])
            elif not self.adv_init:
                adv_input =0
        pro_reward, adv_error = self.pro_adv_apply_action(pro_input, adv_input)
        self.total_steps+=1


        # prepare train batch
        if self.train_pro and not self.is_test and self.use_force_filter: #modified by Yoon
            if self.is_valid_sample:
                if not self.lift_success:
                    if len(self.failed_cases_stack1) < 3:
                        self.failed_cases_stack1.append(self.patch_Is_resized)
                    if len(self.failed_cases_stack2) < 3:
                        self.failed_cases_stack2.append(self.fc8_predictions)
                    else:
                        self.failed_cases_stack1[np.random.randint(0, len(self.failed_cases_stack1))] = self.patch_Is_resized
                        self.failed_cases_stack2[np.random.randint(0, len(self.failed_cases_stack2))] = self.fc8_predictions

                else:
                    self.num_adv_samples += 1

                self.patch_Is_resized_vec2.append(self.patch_Is_resized)
                self.y_train_vec2.append(pro_reward)
                self.fc8_predictions_vec2.append(self.fc8_predictions)  # [1,1]
                self.pro_train_nums_filter += 1
            
            else:
                self.num_help_samples += 1

            if self.total_steps == self.max_steps:
                while not (self.train_pro and self.pro_train_nums_filter % self.train_batch_update == 0 and not self.is_test):
                    self.patch_Is_resized_vec2.append(self.failed_cases_stack1[-1])
                    self.y_train_vec2.append(0)
                    self.fc8_predictions_vec2.append(self.failed_cases_stack2[-1])  # [1,1]
                    self.pro_train_nums_filter += 1


        if self.train_pro and not self.is_test: #modified by Yoon

            self.patch_Is_resized_vec.append(self.patch_Is_resized)
            self.y_train_vec.append(pro_reward)
            self.fc8_predictions_vec.append(self.fc8_predictions)  # [1,1]
            self.pro_train_nums += 1
            print("failed cases: ", len(self.failed_cases_stack1))
            if self.total_steps == self.max_steps:

                while (not (self.train_pro and self.pro_train_nums % self.train_batch_update == 0 and not
                self.is_test)) and self.use_force_filter:
                    self.patch_Is_resized_vec.append(self.failed_cases_stack1[-1])
                    self.y_train_vec.append(0)
                    self.fc8_predictions_vec.append(self.failed_cases_stack2[-1])  # [1,1]
                    self.pro_train_nums += 1



        print(colored('current adv_batch_size: {}, pro_batch_size: {}'.format(self.train_nums, self.pro_train_nums), 'blue'))
        ### train protagonist, update every batch size

        print("opt : ", self.option)
        #print("step - total : ", self.total_steps)
        #print("train pro : ", self.train_pro)
        #print("pro_train_nums : ", self.pro_train_nums)
        #print("max step: ", self.max_steps)
        #print("adv num: ", self.num_adv_samples)
        #print("help num: ", self.num_help_samples)

        if self.train_pro and self.pro_train_nums % self.train_batch_update == 0 and not self.is_test: #modified by Yoon
            print("train G . . .")
            patch_Is_resized_vec_in = np.vstack(self.patch_Is_resized_vec)
            y_train_vec_in = np.array(self.y_train_vec)[np.newaxis, :].transpose()
            fc8_predictions_vec_in = np.array(self.fc8_predictions_vec)[np.newaxis, :].transpose()

            print(colored('debug patch_Is shape: {}, y_train_vec_in.shape: {}, prediction: {}'.format(patch_Is_resized_vec_in.shape, y_train_vec_in.transpose(), fc8_predictions_vec_in.transpose()),'red'))
            #yyt
            train_loss = self.G.train_batch(patch_Is_resized_vec_in, y_train_vec_in, self.pro_save_num, model_path_completion('models/pro_model'+self.log_name), self.train_batch_update)

            self.G.saver.save(self.G.sess, model_path_completion(
                'models/pro_model' + self.log_name + '-' + str(self.total_steps)) + "-" + str(self.option))

            print(colored('new protagonist policy saved at: {}'.format(model_path_completion('models/pro_model' + self.log_name + '-'+ str(self.total_steps))),'cyan'))
            with open(self.pro_loss_log_path, 'a') as fw:
                fw.write(str(train_loss) + '\n')


            self.fc8_predictions_vec=[]
            self.patch_Is_resized_vec=[]
            self.y_train_vec=[]
            self.pro_train_nums =1
            self.pro_save_num +=1
            if self.pro_save_num % 6==0:
                self.pro_save_num =1



        if self.train_pro and self.pro_train_nums_filter % self.train_batch_update==0 and not self.is_test and self.use_force_filter: #modified by Yoon
            print("train G_filter . . .")
            patch_Is_resized_vec_in = np.vstack(self.patch_Is_resized_vec2)
            y_train_vec_in = np.array(self.y_train_vec2)[np.newaxis, :].transpose()
            fc8_predictions_vec_in = np.array(self.fc8_predictions_vec2)[np.newaxis, :].transpose()

            print(colored('debug patch_Is shape: {}, y_train_vec_in.shape: {}, prediction: {}'.format(patch_Is_resized_vec_in.shape, y_train_vec_in.transpose(), fc8_predictions_vec_in.transpose()),'red'))

            train_loss = self.G_filter.train_batch(patch_Is_resized_vec_in, y_train_vec_in, self.pro_save_num2, model_path_completion('models/pro_model'+self.log_name), self.train_batch_update)

            self.G_filter.saver.save(self.G_filter.sess, model_path_completion(
                'models/pro_model' + self.log_name + '-' + str(self.total_steps)) + "-force-filter" + "-" + str(
                self.option))

            print(colored('new protagonist with filter policy saved at: {}'.format(model_path_completion('models/pro_model' + self.log_name + '-'+ str(self.total_steps) + "-force-filter" + "-" + str(self.option))),'cyan'))
            with open(self.pro_loss_log_path2, 'a') as fw:
                fw.write(str(train_loss) + '\n')


            self.fc8_predictions_vec2=[]
            self.patch_Is_resized_vec2=[]
            self.y_train_vec2=[]
            self.pro_train_nums_filter =1
            self.pro_save_num2 +=1
            if self.pro_save_num2 % 6==0:
                self.pro_save_num2 =1




        ### train adversary
        if not self.is_human and self.to_train and self.train_nums % self.batch_update==0:

            X_batches = np.vstack(self.rot_images).astype(float)
            y_batch = np.vstack(self.rot_y_batches).astype(float)

            print(colored('perform training with batches!', 'blue'))
            # original: train adv policy
            # adv_loss = train_adv(self.adv_policy, y_batch, self.adv_policy['sess'],X_batches, self.total_steps)
            # save_path = self.adv_policy['saver'].save(self.adv_policy['sess'], model_path_completion('models/adv_model' + self.log_name +'-'+str(self.total_steps) ))

            #debug, shakeNet, train adv policy
            print(colored('debug adv_batch_shape: {}, adv_y_batch: {}'.format(X_batches.shape, y_batch.shape),'cyan'))
            adv_loss = self.G_adv.train_batch(X_batches, y_batch, self.batch_update)
            self.G_adv.saver.save(self.G_adv.sess, model_path_completion('models/adv_model' + self.log_name + '-' +str(self.total_steps)))
            print(colored('new adversarial policy saved at: {}'.format(model_path_completion('models/adv_model' + self.log_name + '-' + str(self.total_steps)))))            
            with open(self.adv_loss_log_path, 'a') as fw:
                fw.write(str(adv_loss) +  '\n')

            # original used
            # self.save_num +=1
            # if self.to_train and self.save_num % 5 == 0 and not self.is_human:
            #     save_path = self.adv_policy['saver'].save(self.adv_policy['sess'], model_path_completion('models/adv_model'+self.log_name))
            #     print(colored('new adversarial policy saved at: {}'.format(save_path), 'blue'))
            #     self.save_num =1


            self.train_nums = 1
            self.rot_images = []
            self.rot_y_batches =[]


        self.write_log_num +=1
        if self.write_log_num % self.write_log_update ==0 or self.total_steps ==50: #-> 200
            # write logs
            if not self.is_human:
                print(colored('record logs for reward/adv_error','green'))
                with open(self.error_log_path, 'a') as fw:
                    # step, adv_error, adv_action
                    for item in self.adv_error_logs:
                        fw.write(str(item[0])+ ' ' + str(item[1]) + ' ' + str(item[2]) + ' '+ item[3] + '\n')

                with open(self.reward_log_path,'a') as fw:
                    for item in self.reward_logs:
                        fw.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + '\n')
            else:
                print(colored('record logs for human reward/adv_error', 'green'))
                with open(self.human_error_log_path, 'a') as fw:
                    # step, adv_error, adv_action
                    for item in self.adv_error_logs:
                        fw.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' '+ item[3]+ '\n')

                with open(self.human_reward_log_path, 'a') as fw:
                    for item in self.reward_logs:
                        fw.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + '\n')


            self.adv_error_logs = []
            self.reward_logs =[]
            self.write_log_num = 1

        if self.total_steps == self.early_stop_num:
            if self.to_train:
                # save_path = self.adv_policy['saver'].save(self.adv_policy['sess'],
                #                                           model_path_completion('models/adv_model' + self.log_name + '-'+str(self.total_steps) ))
                self.G_adv.saver.save(self.G_adv.sess, model_path_completion('models/adv_model'+self.log_name+'-'+str(self.total_steps)))
            if self.train_pro:
                self.G.saver.save(self.G.sess, model_path_completion('models/pro_model' + self.log_name + '-'+ str(self.total_steps) ))


        info={}
        info['steps']=self.total_steps
        info['adv_forces'] = self.adv_forces
        info['idx_to_action']=self.idx_to_action

        return self._get_observation(), pro_reward, adv_error, 1, info


    # originally used for checking height success after perturbation
    def _check_success3(self, sim):
        cube_height = sim.data.body_xpos[self.cube_body_id][2]
        table_height = sim.data.body_xpos[self.table_body_id][2]
        print(colored('after -> table_height: {}, cube_height: {}'.format(table_height, cube_height),'blue'))
        return cube_height > table_height + self.geom_x_new + 0.30


    # used for checking height success before perturbation
    def _check_success2(self, sim):

        cube_height = sim.data.body_xpos[self.cube_body_id][2]
        #todo
        table_height = sim.data.body_xpos[self.table_body_id][2]

        print(colored('before -> table_height: {}, cube_height: {}'.format(table_height, cube_height),'red'))

        return cube_height > table_height + 0.30



    def _adv_check3(self, sim):
        nut_id= sim.model.body_name2id('object0')
        object_pos=sim.data.body_xpos[nut_id]
        dis = np.linalg.norm(object_pos-self.ori_nut_pos)
        thresh = 1
        print(colored('debug: dis: {}'.format(dis), 'yellow'))
        if dis > thresh:
            return True
        else:
            return False




    # not grasp, true
    def  _adv_check2(self,sim):
        touch_left_finger = False
        touch_right_finger = False

        self.cube_geom_ids = [sim.model.geom_name2id('object0')]
        if self.object_xml == 'round-nut.xml':
            round_nut_ids = [sim.model.geom_name2id('object{}'.format(i)) for i in range(0, 9)]
            self.cube_geom_ids = round_nut_ids
        if self.object_xml == 'half-nut.xml':
            half_nut_ids = [sim.model.geom_name2id('object{}'.format(i)) for i in range(0,6)]
            self.cube_geom_ids = half_nut_ids

        if self.object_xml == 'new_cube.xml':
            new_cube_ids = [sim.model.geom_name2id('object{}'.format(i)) for i in range(0,2)]
            self.cube_geom_ids = new_cube_ids
        
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        cube_contact_table=[]
        for i in range(len(sim.data.contact)):
            c = sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 in self.cube_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.cube_geom_ids and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 in self.cube_geom_ids:
                touch_right_finger = True
            if c.geom1 in self.cube_geom_ids and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        print("touch left: ", touch_left_finger)
        print("touchright: ", touch_right_finger)
        if touch_right_finger and touch_left_finger:
            return False
        else:
            return True

    # not grasp, true
    def _adv_check_middle(self, sim):
        touch_left_finger = False
        touch_right_finger = False

        self.cube_geom_ids = [sim.model.geom_name2id('object0')]
        if self.object_xml == 'round-nut.xml':
            round_nut_ids = [sim.model.geom_name2id('object{}'.format(i)) for i in range(0, 9)]
            self.cube_geom_ids = round_nut_ids
        if self.object_xml == 'half-nut.xml':
            half_nut_ids = [sim.model.geom_name2id('object{}'.format(i)) for i in range(0, 6)]
            self.cube_geom_ids = half_nut_ids

        if self.object_xml == 'new_cube.xml':
            new_cube_ids = [sim.model.geom_name2id('object{}'.format(i)) for i in range(0, 2)]
            self.cube_geom_ids = new_cube_ids

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        cube_contact_table = []
        for i in range(len(sim.data.contact)):
            c = sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 in self.cube_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.cube_geom_ids and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 in self.cube_geom_ids:
                touch_right_finger = True
            if c.geom1 in self.cube_geom_ids and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        print("touch left: ", touch_left_finger)
        print("touchright: ", touch_right_finger)
        if touch_right_finger or touch_left_finger:
            return False
        else:
            return True



    def _pre_action(self, action):
        # action is supposed to be number of joints
        self.sim.data.ctrl[:] = action



    def _post_action(self, action):
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, {}




    def reward(self, action=None):
        return 0



    def render(self):
        self.viewer.render()




    def observation_spec(self):

        observation = self._get_observation()
        return observation



    def action_spec(self):
        raise NotImplementedError





    def _destroy_viewer(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.viewer.window)
            self.viewer = None



    def close(self):
        self._destroy_viewer()


    def close_all(self):
        if self.G is not None:
            self.G.sess.close()
            del self.G

        if self.G_filter is not None:
            self.G_filter.sess.close()
            del self.G_filter


    def set_speed(self, speed):
        self.speed = speed



    def set_pert(self, pert):
        self.pert = pert

