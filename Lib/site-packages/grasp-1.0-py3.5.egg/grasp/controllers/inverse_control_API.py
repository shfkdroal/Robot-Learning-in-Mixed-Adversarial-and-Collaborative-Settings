from .inverse_kinematics import inverse_kinematics
import numpy as np
import copy
import os
import math
import mujoco_py.cymj as cymj
from mujoco_py.generated import const
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions

from grasp.utils.mjcf_utils import xml_path_completion

def control_from_pts(pt, quat, model_name):
    target_pos1= np.array([0.78, 0.0, 1.1])
    target_pos2= np.append(pt, [0.78])
    target_quat = np.append(quat, [0])

    quat_z = np.array([math.cos(math.pi/4),0.0,0.0, math.sin(math.pi/4)])
    new_quat = np.empty(4)
    cymj._mju_mulQuat(new_quat, quat_z, target_quat)

    qpos1= inverse_kinematics(model_name, body_name='left_gripper_base',
                              target_pos = target_pos1, target_quat= new_quat, image_name='a')
    qpos=[]
    num=5
    for n in range(1, num):
        cpos = target_pos1+(target_pos2-target_pos1)/num*n
        qpos.append(inverse_kinematics(model_name, body_name='left_gripper_base',
                                      target_pos = cpos, target_quat=new_quat, image_name='a'))

    model = load_model_from_path(model_name)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.pert.select = 1
    max_step=5000
    step=1
    idx =[]
    for name in sim.model.actuator_names:
        idx.append(sim.model.get_joint_qpos_addr(name))

    # viewer.cam.fixedcamid = 0
    # viewer.cam.type = const.CAMERA_FIXED

    # add gravity
    object_id = sim.model.body_name2id('object0')
    sim.data.xfrc_applied[object_id][2] = -1

    print('move to qpos1')
    sim.data.ctrl[:] = qpos1[idx]
    sim.data.ctrl[16] = 0.2
    sim.data.ctrl[17] = -0.2

    while step <= max_step / 2:
        sim.step()
        viewer.render()
        step += 1

    # slowly move to qpos2
    print('slowly move to qpos2')
    for n in range(0, num - 1):
        sim.data.ctrl[:] = qpos[n][idx]
        sim.data.ctrl[16] = 0.2
        sim.data.ctrl[17] = -0.2
        step = 1
        while step <= max_step / 10:
            sim.step()
            viewer.render()
            step += 1

    print('grasp object')
    sim.data.ctrl[16] = -0.2
    sim.data.ctrl[17] = 0.2
    step = 1
    while step <= max_step / 3:
        sim.step()
        viewer.render()
        step += 1

    # if grasping succeeds, apply perturbation
    print('lift object')
    for n in range(num - 2, -1, -1):
        sim.data.ctrl[:] = qpos[n][idx]
        sim.data.ctrl[16] = -0.2
        sim.data.ctrl[17] = 0.2
        step = 1
        while step <= max_step / 10:
            sim.step()
            viewer.render()
            step += 1

    # sim.data.xfrc_applied[1][1]=1

    # while True:
    #     sim.step()
    #     viewer.render()



def pro_adv_apply_action(pro_action, adv_action):
    model_name = xml_path_completion('Baxter/baxter/master.xml')
    control_from_pts(pro_action[0][:2], pro_action[1], model_name)
