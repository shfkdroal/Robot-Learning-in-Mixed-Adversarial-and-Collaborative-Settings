import cv2
import numpy as np
from .grasp_learner import grasp_obj
from .grasp_predictor import Predictors
from .shake_learner import shake_obj
from .shake_predictor import Predictors as Adv_Predictors
import time
import os
import ipdb

from grasp.utils.mjcf_utils import xml_path_completion
from grasp.utils.mjcf_utils import image_path_completion
from grasp.utils.mjcf_utils import root_path_completion
from grasp.utils.mjcf_utils import root_path_completion2
from grasp.utils.mjcf_utils import predict_image_path_completion

from mujoco_py import MjSim, MjViewer, load_model_from_path, functions
from mujoco_py.generated import const

from termcolor import colored


def drawRectangle(I, h, w, t, gsize=300):
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
    im_points = rot_points + np.array([w,h])
    # print('rec points: ',im_points)

    min_x = min(im_points[0][0], im_points[2][0], im_points[2][0], im_points[3][0])
    max_x = max(im_points[0][0], im_points[1][0], im_points[2][0], im_points[3][0])
    min_y = min(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
    max_y = max(im_points[0][1], im_points[1][1], im_points[2][1], im_points[3][1])
    center_pt = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    print('center points: ', center_pt)
    cv2.circle(I_temp, center_pt, 2, (255, 0, 0), -1)

    cv2.line(I_temp, tuple(im_points[0].astype(int)), tuple(im_points[1].astype(int)), color=(0,255,0), thickness=5)
    cv2.line(I_temp, tuple(im_points[1].astype(int)), tuple(im_points[2].astype(int)), color=(0,0,255), thickness=5)
    cv2.line(I_temp, tuple(im_points[2].astype(int)), tuple(im_points[3].astype(int)), color=(0,255,0), thickness=5)
    cv2.line(I_temp, tuple(im_points[3].astype(int)), tuple(im_points[0].astype(int)), color=(0,0,255), thickness=5)
    return (I_temp, center_pt, R, grasp_angle)




# if train: root_path_completion
# if inference: root_path_completion2
def init_detector(num_samples,use_pro_new='False', use_pro_name='', batch_size=1, gpu_id=0, lr_rate=0.01, test_user= False):
    # init
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

    print('Loading grasp model')
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



# nbatches is batch size
def predict_from_img(image, obj_name, G, timestep, log_dir, is_train=True):
    gscale = 0.234
    I = cv2.imread(image)
    imsize = 896
    gsize = int(gscale * imsize)

    # detector needs to have been initialized
    P = Predictors(I, G)

    print('Predicting on samples')
    st_time = time.time()

    print(colored('is_train in predict_from_img: {}'.format(is_train),'red'))
    P.graspNet_grasp(patch_size=gsize, num_samples=G.BATCH_SIZE, is_train = is_train)
    #P.graspNet_grasp(patch_size=gsize, num_samples=1000)
    print('Time taken: {}s'.format(time.time() - st_time))

    print('debug predict_from_img fc8_norm_vals shape: ', P.fc8_norm_vals.shape)

    if not is_train:
        index = np.argmax(P.fc8_norm_vals)
        r_pindex = index // P.fc8_norm_vals.shape[1]
        r_tindex = index%P.fc8_norm_vals.shape[1]
    else:
        r_pindex = P.r_pindex
        r_tindex = P.r_tindex


    (I, center_pt, R, angle) = drawRectangle(I, P.patch_hs[r_pindex], P.patch_ws[r_pindex], r_tindex, gsize)


    patch_Is_resized = P.patch_Is_resized[r_pindex]
    patch_Is_resized = patch_Is_resized[np.newaxis, :] #[1,224,224,3]

    # save_name = image_path_completion(obj_name + '_predict.jpg')
    save_name = predict_image_path_completion(obj_name + '_predict_{}.jpg'.format(timestep), log_dir)
    try:
        cv2.imwrite(save_name, I)
        print('prediction image {} saved'.format(save_name))
    except:
        print('error saving image: {}'.format(save_name))
    
    # G.test_close()
    pred = P.pred
    pred = np.max(pred) #[1]
    del P
    # one patch, one prediction [1, 18], one groundtruth [1,18]
    return (center_pt, R, angle, patch_Is_resized, pred)




# debug, adv_precition using shakeNet
def adv_predict_from_img(I, G_adv):
    A = Adv_Predictors(I, G_adv)
    print('Adv Predicting on samples')
    st_time = time.time()
    probs, adv_action = A.shakeNet_shake(num_actions= 6, num_samples = 1)
    print('Time take: {}s'.format(time.time()-st_time))
    del A

    return probs,adv_action
