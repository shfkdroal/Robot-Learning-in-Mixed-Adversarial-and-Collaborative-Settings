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
from grasp.predict_module import init_detector
from grasp.predict_module import build_network
from grasp.predict_module import train_adv
from grasp.predict_module import prepare_X_batch
from grasp.predict_module import prepare_X_batch2
# debug
from grasp.predict_module import adv_predict_from_img
from grasp.predict_module import init_adv


import os
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
        self.train_batch_update=10

        print(colored('initialize protagonist detection model', 'red'))
        self.lr_rate =0.001
        if not self.random_init:
            if self.object_xml =='half-nut.xml' :
                self.lr_rate = 0.00001
            if self.object_xml=='round-nut.xml':
                self.lr = 0.005
        self.G= init_detector(self.num_samples, use_pro_new, use_pro_name, self.num_samples, gpu_id=0, lr_rate = self.lr_rate, test_user = test_user) #batchsize =1

        # the first means no force is applied
        # object to use
        force=3.5
        if not self.random_init:
            force = 3.5
            if self.object_xml=='bottle.xml':
                force=3.5
            if self.object_xml=='bread.xml':
                force=3
            if self.object_xml=='new_cube.xml':
                force=3.0
            if self.object_xml=='round-nut.xml':
                force=6 #later changed to 4
            if self.object_xml =='half-nut.xml':
                force = 5.5
        # if self.object_xml =='cube.xml':
            # force = 3.0

        down_force = 0.3
        if not self.random_init:
            down_force = 1
            if self.object_xml == 'cube.xml':
                down_force = 0.3
            if self.object_xml == 'new_cube.xml':
                down_force = 0.5
            if self.object_xml =='half-nut.xml':
                down_force= 0.3

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

        # collect adv inputs
        self.rot_images=[]
        self.rot_y_batches=[]
        self.train_nums =1
        self.pro_train_nums=1
        self.save_num =1
        self.pro_save_num =1


        # record reward, adv_error
        self.adv_error_logs=[]
        self.reward_logs=[]

        self.total_steps = 0

        self.use_render = use_render
        self.use_new_model = use_new_model
        self.to_train = to_train

        self.error_log_path = log_path_completion('error_log{}.txt'.format(self.log_name))
        self.reward_log_path = log_path_completion('reward_log{}.txt'.format(self.log_name))
        self.adv_loss_log_path = loss_path_completion('adv_loss_log{}.txt'.format(self.log_name))
        self.pro_loss_log_path = loss_path_completion('pro_loss_log{}.txt'.format(self.log_name))
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
        np.random.seed(self.seed)



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



    def get_ik_param(self, object_xml):
        print(colored('Init scene with top-down camera','red'))
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
        imageio.imwrite(image_path_completion('crop.jpg'), crop_frame)

        (center_pt, R, grasp_angle, patch_Is_resized, fc8_predictions) = predict_from_img(image_path_completion('crop.jpg'), object_xml, self.G, self.total_steps, self.log_name, is_train=self.train_pro)
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
        cube_x_new = np.random.uniform(cube_pos_x_low, cube_pos_x_high)
        cube_y_new = np.random.uniform(cube_pos_y_low, cube_pos_y_high)
        root[-1][0].attrib['pos'] = ' '.join([str(i) for i in [cube_x_new, cube_y_new, cube_pos[2]]])

        quat_vec =  root[-1][0].attrib['quat'].strip().split(' ')
        quat_vec =[i for i in quat_vec if i!='']
        body_xquat = [float(i) for i in quat_vec]
        radian = np.random.uniform(low= -45, high=45) * np.pi/180
        new_quat = np.empty(4)
        functions.mju_mulQuat(new_quat, np.array([np.cos(radian/2), 0., 0., np.sin(radian/2)]), np.array(body_xquat))

        root[-1][0].attrib['quat'] = ' '.join([str(i) for i in new_quat])
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
        print(colored('current object: {}'.format(self.object_xml), 'red'))
        if self.random_init:
            self._reset_xml2(self.object_xml)
        else:
            self._reset_xml(self.object_xml)
        self._reset_internal()
        self.sim.forward()

        # debug, get param for ik
        (grasp_pt, new_quat, rot_image, self.patch_Is_resized, self.fc8_predictions) = self.get_ik_param(self.object_xml)
        print('debug reset fc8_predictions shape: ', self.fc8_predictions)

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



    def pro_adv_apply_action(self, pro_action, adv_action=None):

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


        print('grasp object')
        sim.data.ctrl[16] = 0
        sim.data.ctrl[17] = 0
        step =1
        while step <= max_step / num:
            ind = sim.model.body_name2id('object0')
            # sim.data.xfrc_applied[ind][2] = -0.1
            sim.step()
            if self.use_render:
                viewer.render()
            step +=1

        print('lift object')  
        for n in range(num-1, -1 , -1):
            sim.data.ctrl[:] = qpos[n][idx]
            sim.data.ctrl[16] = -0.2
            sim.data.ctrl[17] = 0.2
            step =1
            while step <= max_step/8:
                sim.step()
                if self.use_render:
                    viewer.render()
                step += 1


        # record grasping point
        viewer.vopt.geomgroup[0]=0
        viewer.vopt.geomgroup[1]=0
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
        gravity[2]=-0.1


        



        #increase time of doing nothing
        if self.object_xml =='round-nut.xml' or self.object_xml=='half-nut.xml' or self.object_xml=='new_cube.xml':
            nut_id = sim.model.body_name2id('object0')
            self.ori_nut_pos = copy.deepcopy(sim.data.body_xpos[nut_id])
            sleep(0.1)


        # perform perturbation when it's -1
        # adv_error is 0 zero when it's not lifted (default)
        adv_error = 0.0
        # pre_reward = self.reward2(sim)
        pre_reward = 0.0
        lift_success = self._check_success2(sim)
        # if height of cube satisfies, perform perturbation
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
                while True:

                    if self.is_human and viewer.key_pressed==glfw.KEY_Q:
                        break
                    if not self.is_human and step == max_step/25:
                        break

                    if self.is_human:
                        adv_action = self.human_perturb_action2(viewer)
                        if  adv_action!=0:
                            human_step+=1
                        if human_step == max_step/25:
                            break
                    sim.data.xfrc_applied[ind][:]= self.adv_forces[adv_action]
                    sim.step()
                    if step%500==0:
                        print(colored('perturbation applied: {}'.format(self.idx_to_action[adv_action]), 'yellow'))
                    if self.use_render:
                        viewer.render()
                    step+=1


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
                tmp = np.zeros((1,self.n_adv_outputs))
                # print('tmp: {}, adv_action: {}'.format(tmp, adv_action))
                # tmp[0][adv_action]=1
                # self.rot_y_batches.append(tmp)
                self.rot_y_batches.append(adv_error)
            self.adv_error_logs.append((self.total_steps, adv_error, adv_action, self.idx_to_action[adv_action]))

        ###debug
        if lift_success==False:
            post_reward =pre_reward
        else:
            if adv_error ==0:
                if self.is_human: 
                    self.adv_prob =1
                elif self.random_perturb:
                    self.adv_prob = 1
                elif not self.adv_init:
                    self.adv_prob = 0
                post_reward = pre_reward-self.alpha* self.adv_prob
            else:
                post_reward = pre_reward
        print(colored('pre_reward: {}, post_reward: {}, adv_error: {}'.format(pre_reward, post_reward, adv_error), 'blue'))
        self.reward_logs.append((self.total_steps, pre_reward, post_reward))

        # close viewer of self first, then destroy this viewer
        self._destroy_viewer()
        if viewer is not None:
            glfw.destroy_window(viewer.window)
            viewer = None

        return post_reward, adv_error

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
        elif c== glfw.KEY_RIGHT:
            adv_action = 3
        elif c== glfw.KEY_I:
            adv_action = 4
        elif c==glfw.KEY_O:
            adv_action = 2
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

        # protagonist input param
        grasp_pt = action['grasp_pt']
        new_quat = action['new_quat']
        pro_input = (grasp_pt, new_quat)

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
        if self.train_pro:
            self.patch_Is_resized_vec.append(self.patch_Is_resized)
            self.y_train_vec.append(pro_reward)
            self.fc8_predictions_vec.append(self.fc8_predictions) #[1,1]
            self.pro_train_nums+=1

        print(colored('current adv_batch_size: {}, pro_batch_size: {}'.format(self.train_nums, self.pro_train_nums), 'blue'))
        ### train protagonist, update every batch size
        if self.train_pro and self.pro_train_nums % self.train_batch_update==0:
            patch_Is_resized_vec_in = np.vstack(self.patch_Is_resized_vec)
            y_train_vec_in = np.array(self.y_train_vec)[np.newaxis, :].transpose()
            fc8_predictions_vec_in = np.array(self.fc8_predictions_vec)[np.newaxis, :].transpose()
            print(colored('debug patch_Is shape: {}, y_train_vec_in.shape: {}, prediction: {}'.format(patch_Is_resized_vec_in.shape, y_train_vec_in.transpose(), fc8_predictions_vec_in.transpose()),'red'))

            train_loss = self.G.train_batch(patch_Is_resized_vec_in, y_train_vec_in, self.pro_save_num, model_path_completion('models/pro_model'+self.log_name), self.train_batch_update)
            self.G.saver.save(self.G.sess, model_path_completion('models/pro_model' + self.log_name + '-'+ str(self.total_steps)))
            print(colored('new protagonist policy saved at: {}'.format(model_path_completion('models/pro_model' + self.log_name + '-'+ str(self.total_steps) )),'cyan'))
            with open(self.pro_loss_log_path, 'a') as fw:
                fw.write(str(train_loss) + '\n')

            self.fc8_predictions_vec=[]
            self.patch_Is_resized_vec=[]
            self.y_train_vec=[]
            self.pro_train_nums =1
            self.pro_save_num +=1
            if self.pro_save_num % 6==0:
                self.pro_save_num =1



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
        if self.write_log_num % self.write_log_update ==0 or self.total_steps ==50:
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
        self.G.sess.close()
        del self.G


    def set_speed(self, speed):
        self.speed = speed



    def set_pert(self, pert):
        self.pert = pert

