import os
import numpy as np
import cv2
import scipy

import grasp


def xml_path_completion(xml_path):

    if xml_path.startswith('/'):
        full_path = xml_path

    else:
        full_path = os.path.join(grasp.models.assets_root, xml_path)

    return full_path



def image_path_completion(image_path):
    if image_path.startswith('/'):
        full_path = image_path

    else:
        full_path = os.path.join('/home/icaros/grasp/training/logs/images', image_path)

    return  full_path


def predict_image_path_completion(image_path, log_dir):
    if image_path.startswith('/'):
        full_path = image_path
    else:
        dir_path = '/home/icaros/grasp/training/logs/images/{}'.format(log_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        full_path = os.path.join(dir_path, image_path)

    return full_path



def root_path_completion(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp/grasp', path)

    return full_path



def root_path_completion2(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp_fixed/grasp', path)

    return full_path


def model_path_completion(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp/grasp/predict_module/', path)
    return full_path


def log_path_completion(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp/training/logs/logs', path)
    return full_path


def human_path_completion(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp/training/logs/human', path)
    return full_path




def loss_path_completion(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp/training/logs/loss', path)
    return full_path



def config_path_completion(path):
    if path.startswith('/'):
        full_path = path
    else:
        full_path = os.path.join('/home/icaros/grasp/training/logs/config', path)
    return full_path




def preprocess(im):
    im = im.astype(float)
    im = cv2.resize(im,(224,224), interpolation=cv2.INTER_CUBIC)
    im = im - 111
    im = im/144 # To scale from -1 to 1
    return im


def rotateImageAndExtractPatch(img, angle, center, size):
    angle = angle*180/np.pi
    padX = [img.shape[1] - center[1], center[1]]
    padY = [img.shape[0] - center[0], center[0]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant')
    imgR = scipy.misc.imrotate(imgP, angle)
    half_size = int(size/2)
    return imgR[padY[0] + center[0] - half_size: padY[0] + center[0] + half_size, padX[0] + center[1] - half_size : padX[0] + center[1] + half_size, :]