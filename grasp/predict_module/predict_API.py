import cv2
import numpy as np
from .grasp_learner import grasp_obj
from .grasp_predictor import Predictors
from .shake_learner import shake_obj
from .force_filter_learner import force_filter_obj
from .five_filter_learner import five_filter_obj
from .shake_predictor import Predictors as Adv_Predictors
from .grasp_learner_with_filter import grasp_obj2

import time
import os
import ipdb
import gc

from grasp.utils.mjcf_utils import xml_path_completion
from grasp.utils.mjcf_utils import image_path_completion
from grasp.utils.mjcf_utils import root_path_completion
from grasp.utils.mjcf_utils import root_path_completion2
from grasp.utils.mjcf_utils import predict_image_path_completion

from mujoco_py import MjSim, MjViewer, load_model_from_path, functions
from mujoco_py.generated import const

from termcolor import colored


def drawRectangle(I, h, w, t, gsize=300, pred=None):
    I_temp = I
    grasp_l = gsize/2.5
    grasp_w = gsize/5.0
    grasp_angle = t*(np.pi/18)-np.pi/2

    points = np.array([[-grasp_l, -grasp_w],
                       [grasp_l, -grasp_w],
                       [grasp_l, grasp_w],
                       [-grasp_l, grasp_w]])
    R = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                  [np.sin(grasp_angle), np.cos(grasp_angle)]])
    rot_points = np.dot(R, points.transpose()).transpose()
    im_points = rot_points + np.array([w, h])
    # print('rec points: ',im_points)

    min_x = min(im_points[0][0], im_points[2][0], im_points[2][0], im_points[3][0])
    max_x = max(im_points[0][0], im_points[1][0], im_points[2][0], im_points[3][0])
    min_y = min(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
    max_y = max(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
    center_pt = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    #print('center points: ', center_pt)
    cv2.circle(I_temp, center_pt, 2, (255, 0, 0), -1)

    cv2.line(I_temp, tuple(im_points[0].astype(int)), tuple(im_points[1].astype(int)), color=(0, 255, 0), thickness=5)
    cv2.line(I_temp, tuple(im_points[1].astype(int)), tuple(im_points[2].astype(int)), color=(0, 0, 255), thickness=5)
    cv2.line(I_temp, tuple(im_points[2].astype(int)), tuple(im_points[3].astype(int)), color=(0, 255, 0), thickness=5)
    cv2.line(I_temp, tuple(im_points[3].astype(int)), tuple(im_points[0].astype(int)), color=(0, 0, 255), thickness=5)


    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = center_pt
    fontScale = 0.5
    fontColor = (255, 0, 0)
    lineType = 2

    cv2.putText(I_temp, str(pred),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    return (I_temp, center_pt, R, grasp_angle)



def drawRectangle_spot(gripper_coords, I, h, w, t, gsize=300, pred=None):
    I_temp = I
    grasp_l = gsize/2.5
    grasp_w = gsize/5.0
    points = np.array([[-grasp_l, -grasp_w],
                       [grasp_l, -grasp_w],
                       [grasp_l, grasp_w],
                       [-grasp_l, grasp_w]])

    grasp_angles = t*(np.pi/18)-np.pi/2

    R = []
    for e in grasp_angles:
        R.append(np.array([[np.cos(e), -np.sin(e)], [np.sin(e), np.cos(e)]]))
    rot_points = []
    for e in R:
        rot_points.append(np.dot(e, points.transpose()).transpose())

    for i, e in enumerate(rot_points):
        im_points = e + np.array([h[i],w[i]])
        # print('rec points: ',im_points)

        min_x = min(im_points[0][0], im_points[2][0], im_points[2][0], im_points[3][0])
        max_x = max(im_points[0][0], im_points[1][0], im_points[2][0], im_points[3][0])
        min_y = min(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
        max_y = max(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
        center_pt = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
        # print('center points: ', center_pt)
        cv2.circle(I_temp, center_pt, 3, (255, 0, 0), -1)


    cv2.line(I_temp, tuple(gripper_coords[0]), tuple(gripper_coords[1]), color=(0, 0, 255),
                 thickness=4)
    cv2.line(I_temp, tuple(gripper_coords[2]), tuple(gripper_coords[3]), color=(0, 0, 255),
                 thickness=4)

    return I_temp


def drawRectangle_spot_on_centre(adv_action, gripper_coords, obj_name,
                                 timestep, log_dir, I, C, pre_reward, post_reward, method_opt=0, gsize=300):

    I_temp = I
    is_adv = False
    adv_count = 0
    col_count = 0
    differ = 0
    grasp_l = gsize/2.5
    grasp_w = gsize/5.0
    points = np.array([[-grasp_l, -grasp_w],
                       [grasp_l, -grasp_w],
                       [grasp_l, grasp_w],
                       [-grasp_l, grasp_w]])

    for i, e in enumerate(C):

        center_pt = e

        grasp_angles = pre_reward[i][1]
        R = np.array([[np.cos(grasp_angles), -np.sin(grasp_angles)], [np.sin(grasp_angles), np.cos(grasp_angles)]])
        rot_point = np.dot(R, points.transpose()).transpose()
        im_points = rot_point + np.array([center_pt[1], center_pt[0]])  # center_pt
        min_x = min(im_points[0][0], im_points[2][0], im_points[2][0], im_points[3][0])
        max_x = max(im_points[0][0], im_points[1][0], im_points[2][0], im_points[3][0])
        min_y = min(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
        max_y = max(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
        center_pt = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))

        differ_spot = post_reward[i][0] - pre_reward[i][0]  # pre_reward - post_reward
        color_tutple = None
        print(colored("differ!:{}".format(differ_spot), "yellow"))
        if differ_spot > 0:#0.0005:
            col_count += 1
            #print(colored("col_force", "yellow"))
            color = int(9000 * differ_spot)
            if color >= 255:
                color = 255
            color_tutple = (color, 0, 0)
        else:
            adv_count += 1
            #print(colored("adv_force", "yellow"))
            color = int(-9000 * differ_spot)
            if color >= 255:
                color = 255
            color_tutple = (0, 0, color)

        cv2.circle(I_temp, center_pt, 3, color_tutple, -1)
        differ += post_reward[i][0] - pre_reward[i][0]  # pre_reward - post_reward

    if method_opt == 0:
        if col_count <= adv_count:
            is_adv = True
    else:
        if differ <= 0:
            is_adv = True


    cv2.line(I_temp, tuple(gripper_coords[0]), tuple(gripper_coords[1]), color=(0, 0, 255),
                 thickness=4)
    cv2.line(I_temp, tuple(gripper_coords[2]), tuple(gripper_coords[3]), color=(0, 0, 255),
                 thickness=4)


    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale = 0.6
    lineType = 2

    force_direction = None
    if adv_action == 1:
        force_direction = "   <-Left"
    elif adv_action == 2:
        force_direction = "   V Down"
    elif adv_action == 3:
        force_direction = "   Right->"
    elif adv_action == 4:
        force_direction = "   ^ Up"

    print(colored(force_direction, "yellow"))

    if is_adv:
        is_adv_string = "Adv "
        fontColor = (0, 0, 255)
    else:
        is_adv_string = "Col "
        fontColor = (255, 0, 0)

    if method_opt == 0:
        is_adv_string = is_adv_string + "adv_count: {}".format(adv_count) + " col_count: {}".format(col_count)
    elif method_opt == 1:
        is_adv_string = is_adv_string + "adv_count: {}".format(adv_count) + " col_count: {}".format(col_count) + \
                        "differ sum: {}".format(differ)

    is_adv_string = is_adv_string + force_direction
    cv2.putText(I_temp, is_adv_string,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    trial_types = "2Perturbed"
    save_name = predict_image_path_completion(obj_name + '_predict_{}_{}.jpg'.format(timestep, trial_types),
                                              log_dir)
    try:
        cv2.imwrite(save_name, I_temp)
        print('prediction image {} saved'.format(save_name))
    except:
        print('error saving image: {}'.format(save_name))

    return is_adv




# if train: root_path_completion
# if inference: root_path_completion2
def init_detector(num_samples, use_pro_new='False', use_pro_name='', batch_size=1, gpu_id=0, lr_rate=0.01, test_user=False):
    # init

    use_pro_name = 'bonus-test6-350-base'  #198'

    if test_user:
        if use_pro_new and os.path.exists(root_path_completion2('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion2('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion2('predict_module/models/Grasp_model')
    else:
        if use_pro_new and os.path.exists(root_path_completion('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion('predict_module/models/Grasp_model')

    #model_path = '/home/icaros/grasp/grasp/predict_module/models/pro_model' + use_pro_name
    print('Loading grasp model')
    print('model-path: ', model_path)

    G = grasp_obj(model_path, gpu_id, num_samples)
    G.BATCH_SIZE = batch_size
    G.test_init(lr_rate)

    return G

def init_detector_with_filter(num_samples,use_pro_new='False', use_pro_name='', batch_size=1, gpu_id=0, lr_rate=0.01, test_user= False):
    # init

    use_pro_name = 'bonus-test6-350-base'  #198'

    if test_user:
        if use_pro_new and os.path.exists(root_path_completion2('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion2('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion2('predict_module/models/filter/Grasp_model')
    else:
        if use_pro_new and os.path.exists(root_path_completion('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion('predict_module/models/filter/Grasp_model')

    #model_path = '/home/icaros/grasp/grasp/predict_module/models/pro_model' + use_pro_name
    print('Loading grasp model')
    print('model-path: ', model_path)

    G = grasp_obj2(model_path, gpu_id, num_samples)
    G.BATCH_SIZE = batch_size
    G.test_init(lr_rate)

    return G


def init_detector_test(num_samples, use_pro_new='False', use_pro_name='', batch_size=1, gpu_id=0, lr_rate=0.01,
                  test_user=False, option=0):
    # init

    use_pro_name = 'bonus-test6-100-' + str(option)  #343'


    #'bonus-test6-198-force-filter'

    if test_user:
        if use_pro_new and os.path.exists(
                root_path_completion2('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion2('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion2('predict_module/models/Grasp_model')
    else:
        if use_pro_new and os.path.exists(
                root_path_completion('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion('predict_module/models/Grasp_model')

    model_path = '/home/icaros/grasp/grasp/predict_module/models/pro_model' + use_pro_name
    print('Loading grasp model')
    print('model-path: ', model_path)

    G = grasp_obj(model_path, gpu_id, num_samples)
    G.BATCH_SIZE = batch_size
    G.test_init(lr_rate)

    return G


def init_detector_test_use_filter(num_samples, use_pro_new='False', use_pro_name='', batch_size=1, gpu_id=0, lr_rate=0.01,
                  test_user=False, option = 0):
    # init

    use_pro_name = 'bonus-test6-100-force-filter-' + str(option)  #198 #96 #95 #---

    use_pro_new = False

    #'bonus-test6-198-force-filter'
    if test_user:
        if use_pro_new and os.path.exists(
                root_path_completion2('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion2('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion2('predict_module/models/Grasp_model')
    else:
        if use_pro_new and os.path.exists(
                root_path_completion('predict_module/models/pro_model{}.index'.format(use_pro_name))):
            model_path = root_path_completion('predict_module/models/pro_model{}'.format(use_pro_name))
        else:
            model_path = root_path_completion('predict_module/models/Grasp_model')

    model_path = '/home/icaros/grasp/grasp/predict_module/models/pro_model' + use_pro_name
    print('Loading grasp model')
    print('model-path: ', model_path)


    G = grasp_obj(model_path, gpu_id, num_samples)
    G.BATCH_SIZE = batch_size
    G.test_init(lr_rate)

    return G




# debug
def init_adv(num_samples, use_new_model='False', use_new_name='', batch_size=1, gpu_id=0,adv_lr_rate=0.01):
    if use_new_model and os.path.exists(root_path_completion('predict_module/models/adv_model{}.index'.format(use_new_name))):
        model_path = root_path_completion('predict_module/models/adv_model{}'.format(use_new_name))
    else:
        model_path = root_path_completion('predict_module/models/Shake_model')
    
    print('Loading shake model')
    G= shake_obj(model_path, gpu_id, num_samples)
    G.BATCH_SIZE = batch_size
    G.test_init(adv_lr_rate)
    return G


#added by Yoon

def init_force_filter(batch_size, gpu_id, lr_rate, Collective_dimension): #2324):

    model_path = 'models/checkpoint' #.ckpt-2000'
    G = force_filter_obj(heckpoint_path=model_path, gpu_id=gpu_id, batch_size=batch_size, Collective_dimension=Collective_dimension)
    return G

def init_force_filter2(batch_size, gpu_id, lr_rate, Collective_dimension, opt, is_alt):

    model_path = None

    if is_alt:
        if opt == 0:
            model_path = 'models/bottle_alt/checkpoint'  # .ckpt-2000'
        elif opt == 1:
            model_path = 'models/new_cube_alt/checkpoint'  # .ckpt-2000'
        elif opt == 2:
            model_path = 'models/cube_alt/checkpoint'  # .ckpt-2000'
        elif opt == 3:
            model_path = 'models/half-nut_alt/checkpoint'  # .ckpt-2000'
        elif opt == 4:
            model_path = 'models/round-nut_alt/checkpoint'  # .ckpt-2000'
    else:
        if opt == 0:
            model_path = 'models/bottle/checkpoint'  # .ckpt-2000'
        elif opt == 1:
            model_path = 'models/new_cube/checkpoint'  # .ckpt-2000'
        elif opt == 2:
            model_path = 'models/cube/checkpoint'  # .ckpt-2000'
        elif opt == 3:
            model_path = 'models/half-nut/checkpoint'  # .ckpt-2000'
        elif opt == 4:
            model_path = 'models/round-nut/checkpoint'  # .ckpt-2000'


    G = five_filter_obj(is_alt=is_alt, opt=opt, lr=lr_rate, checkpoint_path=model_path, gpu_id=gpu_id, batch_size=batch_size, Collective_dimension=Collective_dimension)
    return G


# nbatches is batch size
def predict_from_img(gripper_coords, post_reward, num_samples, training_gt, image, obj_name, G, timestep,
                     log_dir, is_train=True, opt=1, update_info=None, was_valid=True):


    print(colored("was valid?: {}".format(was_valid), "yellow"))
    gscale = 0.234
    I = cv2.imread(image)
    imsize = 896
    gsize = int(gscale * imsize)


    # detector needs to have been initialized
    P = Predictors(I, G)
    P.opt = opt
    P.update_info = update_info


    print('Predicting on samples')
    st_time = time.time()

    print(colored('is_train in predict_from_img: {}'.format(is_train), 'red'))


    print("gsize: ", gsize)

    P.graspNet_grasp(patch_size=gsize, num_samples=num_samples) #, is_train = is_train)


    if not (opt == 2):
        epsil = (post_reward + 1) / 2
        if not epsil == 1:
            epsil = (epsil + 0.4) / 2

        #cointoss = np.random.choice(2, 1, p=[1-epsil, epsil])
        cointoss = np.random.choice(2, 1, p=[epsil, 1-epsil])

        # P.graspNet_grasp(patch_size=gsize, num_samples=1000)
        print('Time taken: {}s'.format(time.time() - st_time))

        np.random.seed(np.random.randint(50))
        Reward_array = P.fc8_norm_vals
        r_pindex = 0  # location
        r_tindex = 0  # angle

        trial_types = None

        if opt == 0:
            if is_train:
                angle_num = Reward_array.shape[1]
                #cointoss = 0 #rl
                if cointoss == 0 or obj_name == 'round-nut.xml':# and was_valid:
                    # print('debug predict_from_img fc8_norm_vals shape: ', P.fc8_norm_vals.shape)
                    index = np.argmax(Reward_array)
                    r_pindex = index // angle_num
                    # print("P.fc8_norm_vals.shape[1]: ", P.fc8_norm_vals.shape[1])
                    print("pindex: ", r_pindex)
                    r_tindex = index % angle_num
                    if training_gt:
                        r_tindex = np.random.randint(angle_num)
                    print("tindex: ", r_tindex)
                    print(colored("choose maximum bat", "yellow"))
                    trial_types = "opt"
                else:
                    epsil = np.random.randint(10)
                    if epsil > 3:
                        index = np.argmax(Reward_array)
                        r_pindex = index // angle_num
                        r_tindex = np.random.randint(angle_num)  # P.t_indice[r_pindex]
                        print("pindex: ", r_pindex)
                        print("tindex: ", r_tindex)
                        print(colored("random angle", "yellow"))
                        trial_types = "ra"

                    elif epsil <= 3 and epsil >= 1:
                        r_pindex = np.random.randint(num_samples)  # // angle_num
                        r_tindex = np.argmax(Reward_array[r_pindex])  # % angle_num
                        print("pindex: ", r_pindex)
                        print("tindex: ", r_tindex)
                        print(colored("random loc", "yellow"))
                        trial_types = "rl"

                    else:
                        r_pindex = np.random.randint(num_samples)  # // angle_num
                        r_tindex = np.random.randint(angle_num)  # P.t_indice[r_pindex]
                        print("pindex: ", r_pindex)
                        print("tindex: ", r_tindex)
                        print(colored("fully randomized trials", "yellow"))
                        trial_types = "fr"

            else:
                index = np.argmax(P.fc8_norm_vals)
                r_pindex = index // P.fc8_norm_vals.shape[1]
                print("pindex: ", r_pindex)
                r_tindex = index % P.fc8_norm_vals.shape[1]
                print("tindex: ", r_tindex)

        fc8_values = P.fc8_norm_vals

        print("P.patch_hs[r_pindex]: ", P.patch_hs[r_pindex])
        print("P.patch_ws[r_pindex]: ", P.patch_ws[r_pindex])

        print("P.patch_hs -shape: ", P.patch_hs.shape)
        print("P.patch_ws -shape: ", P.patch_ws.shape)
        # R : Rotational Info

        pred = P.pred

        R_table_update_info = []
        R_table_spec = None
        if opt == 1:
            pred_spot = np.max(pred, axis=1)  # [1]
            r_tindice = np.argmax(pred, axis=1)
            I = drawRectangle_spot(gripper_coords, I, P.patch_hs, P.patch_ws, r_tindice, gsize,
                                                     pred=pred_spot)

            R_table_update_info.append(P.r_angle_table_patches)
            R_table_update_info.append(P.patch_hs)  # 1
            R_table_update_info.append(P.patch_ws)  # 2

            R_table_update_info.append(r_pindex)  # 3
            R_table_update_info.append(r_tindice)  # 4
            R_table_spec = P.R_table_spec
            trial_types = "1Lifted"

        # Added by Yoon for test
        R = None
        angle = None
        center_pt = None
        pred = np.max(pred)
        if opt == 0:
            (I, center_pt, R, angle) = drawRectangle(I, P.patch_hs[r_pindex], P.patch_ws[r_pindex], r_tindex, gsize,
                                                     pred=pred)
            print("center from img: ", center_pt)
        # R : Rotational Info

        patch_Is_resized = P.patch_Is_resized[r_pindex]
        patch_Is_resized = patch_Is_resized[np.newaxis, :]  # [1,224,224,3]

        # save_name = image_path_completion(obj_name + '_predict.jpg')
        save_name = predict_image_path_completion(obj_name + '_predict_{}_{}.jpg'.format(timestep, trial_types),
                                                  log_dir)
        try:
            cv2.imwrite(save_name, I)
            print('prediction image {} saved'.format(save_name))
        except:
            print('error saving image: {}'.format(save_name))

        R_table_spec = P.R_table_spec
        del P
        gc.collect()
        if opt == 1:
            return (patch_Is_resized, pred, R_table_spec, R_table_update_info)
        else: # opt == 0
            return (center_pt, R, angle, patch_Is_resized, pred, fc8_values, R_table_spec, R_table_update_info)

        # modified by Yoon
    else:
        fc8_values = P.fc8_norm_vals
        R_table_spec = P.R_table_spec
        R_table_update_info = []
        R_table_update_info.append(P.r_angle_table_patches)
        R_table_update_info.append(P.patch_hs)
        R_table_update_info.append(P.patch_ws)

        del P

        return R_table_spec, R_table_update_info




def predict_from_R_table(R_table):

    W, H  = R_table.shape
    denom = .0

    max_w = 0
    max_h = 0
    max_num = 0

    for w in range(W):
        for h in range(H):
            e = R_table[w, h]
            p = e[0]
            denom += p

    for w in range(W):
        for h in range(H):
            e = R_table[w, h]
            p = e[0]/denom
            p = [p, 1-p]
            if p[0] >= 0.0:
                sampled = np.random.choice(2, 10, p=p)
                zero_counts = (sampled == 0).sum()
                if max_num < zero_counts or (max_num == zero_counts and np.random.randint(2) == 0):
                    max_w = w
                    max_h = h
                    max_num = zero_counts

    indice = (max_w, max_h)

    print(colored("Reward!: {}".format(R_table[max_w, max_h][0]), "red"))
    print(colored("indices!: {}".format(indice), "red"))

    return (indice, R_table[max_w, max_h][1])



def adv_predict_from_img(I, G_adv):
    A = Adv_Predictors(I, G_adv)
    print('Adv Predicting on samples')
    st_time = time.time()
    probs, adv_action = A.shakeNet_shake(num_actions= 6, num_samples = 1)
    print('Time take: {}s'.format(time.time()-st_time))
    del A

    return probs,adv_action
