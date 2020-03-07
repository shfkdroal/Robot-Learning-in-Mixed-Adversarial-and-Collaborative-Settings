# plot training curve, loss curve
# plot histogram of forces (human , simulated, random_perturb)
# calc success rates for both simulated and human and random_perturb
# plot force selected over time

import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import imageio
import os
import shutil
import pandas as pd

root_path = '/home/icaros/grasp/plot'
log_path = '/home/icaros/grasp/training/logs'

def plot_fig(info, ex_info):
    save_path = os.path.join(root_path, info['log_name'])
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # get paths
    if not info['is_human']:
        error_log_path = os.path.join(log_path, 'logs', 'error_log'+info['log_name']+'.txt')
        reward_log_path = os.path.join(log_path, 'logs', 'reward_log'+info['log_name']+'.txt')

    if info['is_human']:
        error_log_path = os.path.join(log_path, 'human', 'human_error_log'+info['log_name']+'.txt')
        reward_log_path = os.path.join(log_path, 'human', 'human_reward_log' + info['log_name']+'.txt')


    # calc success rate
    rewards= np.loadtxt(reward_log_path)
    grasp_num = rewards.shape[0]
    post_success_num = np.sum(rewards[:,2]==1)
    pre_success_num = np.sum(rewards[:,1]==1)
    success_perturb = pre_success_num - post_success_num
    success_rate = post_success_num/grasp_num
    robust_rate = post_success_num / pre_success_num

    print(colored('total num: {}\npre_success: {}\npost_success: {}\nsuccess_perturb: {}\nsuccess_rate: {}%\nrobust_rate: {}%\n '.format(grasp_num, pre_success_num, post_success_num, success_perturb, success_rate*100, robust_rate*100),'red'))

    with open(os.path.join(save_path, 'result{}.txt'.format(info['log_name'])), 'a') as fw:
        fw.write('grasp_num: '+str(grasp_num) +'\n')
        fw.write('pre_success: '+str(pre_success_num) +'\n')
        fw.write('post_success: '+str(post_success_num)+'\n')
        fw.write('success_perturb: '+str(success_perturb)+'\n')
        fw.write('success_rate: ' + str(success_rate*100) + '%\n')
        fw.write('robust_rate: '+ str(robust_rate*100) + '%\n')


    # plot force histogram
    errors = pd.read_csv(error_log_path, delimiter=' ', names=['time', 'adv_error', 'action', 'action_name'])
    perturbs = errors['adv_error']
    perturb_success = np.sum(perturbs ==0)
    perturb_fail = perturbs.shape[0]-perturb_success

    assert(perturb_success == success_perturb)

    adv_action = errors['action']
    adv_idx_to_action = ex_info['idx_to_action']
    key_num = len(adv_idx_to_action.keys())
    res_vec = []
    for i in range(key_num):
        val= np.sum(adv_action == i)
        res_vec.append(val)

    # plot force histogram
    plt.bar(np.arange(len(res_vec)),height=res_vec)
    plt.xticks(np.arange(len(res_vec)), [adv_idx_to_action[i] for i in range(key_num)])
    plt.xlabel('adversary action')
    plt.ylabel('occurrence')
    plt.title('adversary force histogram')
    save_name = 'adversary histogram '+info['log_name']+'.jpg'
    plt.savefig(os.path.join(save_path, save_name))

    print('debug: res_vec ', res_vec)
    print('force distribution image saved at: {}'.format(os.path.join(save_path, save_name)))

    # plot force distribution over time, with arrow
