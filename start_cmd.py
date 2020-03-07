'''
Train adversary policy
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import grasp
from termcolor import colored
import random
from plot.plot_result import plot_fig


def start_cmd(obj):
    TRAIN_HUMAN=True
    TEST=False
    TRAIN_SELF=False
    TRAIN_ADV = False
    LOG='showing-video'
    OBJECT_XML=obj
    USE_PRO_NEW = False
    USE_PRO_NAME=''
    USE_NEW_NAME=''

    if TRAIN_HUMAN:
        use_render = True
        use_pro_new = False
        is_human = True
        train_pro = True
        random_perturb= False
        use_pro_name = USE_PRO_NAME
        use_new_model = False
        use_new_name = USE_NEW_NAME
        to_train=False
        adv_init = False
    elif TEST:
        use_render = False
        use_pro_new = USE_PRO_NEW
        use_pro_name = USE_PRO_NAME
        is_human = False
        train_pro = False
        random_perturb=True
        use_new_model = False
        use_new_name = USE_NEW_NAME
        to_train = False
        adv_init = False
    elif TRAIN_SELF:
        use_render=False
        use_pro_new= False
        use_pro_name=USE_PRO_NAME
        is_human=False
        train_pro= True
        random_perturb=False
        use_new_model = False
        use_new_name = USE_NEW_NAME
        to_train = False
        adv_init = False
    elif TRAIN_ADV:
        use_render = False
        use_new_model = False
        use_new_name = USE_NEW_NAME
        is_human = False
        train_pro = True
        random_perturb= False
        to_train = True
        adv_init = True
        use_pro_new = False
        use_pro_name = USE_PRO_NAME



    parser= ArgumentParser()
    parser.add_argument('--use_render', default=use_render, help='Set True to render')
    parser.add_argument('--log_name', default=LOG, help='Required for a new log')
    parser.add_argument('--use_new_model', default=use_new_model, help='Set True to use new adversarial model for inference')
    parser.add_argument('--use_new_name', default=use_new_name, help='Adv model to do inference')
    parser.add_argument('--use_pro_new', default=use_pro_new, help='Set True to use new protagonist model for inference')
    parser.add_argument('--use_pro_name', default=use_pro_name, help='Pro model to do inference')
    parser.add_argument('--to_train', default=to_train, help='Set False to disable adversarial training')
    parser.add_argument('--adv_init', default=adv_init, help='Set False to disable adversarial initialization')
    parser.add_argument('--is_human', default=is_human, help='Set True to use human interactive interface')
    parser.add_argument('--train_pro', default=train_pro, help='Set False to disable protagonist training')
    parser.add_argument('--random_perturb', default=random_perturb, help='Set True to use random perturbation for testing')
    parser.add_argument('--object_xml', default=OBJECT_XML, help='Choose object to play with')
    parser.add_argument('--user_name', default="Jiali", help='Specify your name')
    parser.add_argument('--seed', default=51, help='Seed to run on')
    parser.add_argument('--test_user', default= False, help = 'Set true only using model_inference script')

    args = parser.parse_args()
    print(colored('Running info: ', 'red'))
    info = vars(args)
    for key, val in info.items():
        print(colored('{}: {}'.format(key, val),'red'))


    print("Welcome to Icaros grasp suite v{}!".format(grasp.__version__))
    print(grasp.__logo__)


    # enable reward shaping
    # use_ik is crucial, observation should only include image observation
    # to_train: train or inference
    # use_new_model: train from scratch or use trained model
    # if is_human = True, it won't be trained
    random.seed(51)
    env = grasp.make(
            'SawyerLift',
            has_renderer=True,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=500,
            reward_shaping=True,
            use_object_obs=True,
            use_render =args.use_render,
            log_name =args.log_name,
            use_new_model= args.use_new_model,
            use_new_name=args.use_new_name,
            use_pro_new = args.use_pro_new,
            use_pro_name=args.use_pro_name,
            to_train=args.to_train,
            adv_init=args.adv_init,
            is_human = args.is_human,
            train_pro= args.train_pro,
            random_perturb=args.random_perturb,
            object_xml=args.object_xml,
            user_name =args.user_name,
            seed = args.seed,
            params= info,
            test_user = args.test_user,
            )

    # set speed and pert id
    env.set_speed(8.)
    env.set_pert(1)


    obs = env.reset()
    for i in range(1000000):
        obs, pro_reward, adv_error, done, ex_info = env.step(obs)
        if done:
            obs = env.reset()
        if ex_info['steps']==200:
            print('training finished!')
            break

    plot_fig(info, ex_info)
