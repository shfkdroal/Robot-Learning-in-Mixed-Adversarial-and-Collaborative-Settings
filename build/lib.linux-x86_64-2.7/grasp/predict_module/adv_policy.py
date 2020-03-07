from scipy.misc import imresize
import numpy as np
from random import sample
from termcolor import colored
import random
import os
import shutil

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
import matplotlib.image as mpimg

from grasp.utils.mjcf_utils import model_path_completion
from grasp.utils.mjcf_utils import log_path_completion


def prepare_image(image, target_width=299, target_height = 299, max_zoom = 0.2):
    random.seed(48)
    height = image.shape[0]
    width = image.shape[1]
    image_ratio= width/height
    target_image_ratio = target_width/target_height
    crop_vertically = image_ratio<target_image_ratio
    crop_width = width if crop_vertically else int(height *target_image_ratio)
    crop_height = int(width/target_image_ratio) if crop_vertically else height

    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height/resize_factor)

    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0+crop_width
    y1= y0+ crop_height

    image = image[y0:y1, x0:x1]

    if np.random.rand()<0.5:
        image = np.fliplr(image)

    image = imresize(image, (target_width, target_height))

    return image.astype(np.float32)/255



def prepare_image_with_tensorflow(image, target_width = 299, target_height = 299, max_zoom = 0.2):
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)
    crop_height = tf.cond(crop_vertically,
                          lambda: width / target_image_ratio,
                          lambda: height)

    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3])   # 3 = number of channels

    image = tf.random_crop(image, box_size)

    image = tf.image.random_flip_left_right(image)

    image_batch = tf.expand_dims(image, 0)

    image_batch = tf.image.resize_bilinear(image_batch, [target_height, target_width])
    image = image_batch[0] / 255  # back to a single image, and scale the colors from 0.0 to 1.0
    return image



def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)




def build_network(height=299, width=299, channels=3, outputs=36, use_new_model=True, use_new_name='', log_name='1', gpu_id=1):
    if gpu_id==-1:
        dev_name='/cpu:0'
    else:
        dev_name = '/gpu:{}'.format(gpu_id)

    with tf.device(dev_name):
        reset_graph()
        X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
        training = tf.placeholder_with_default(False, shape=[])
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

        inception_saver = tf.train.Saver()

        prelogits = tf.squeeze(end_points['PreLogits'],axis=[1,2])

        n_outputs = outputs

        with tf.name_scope("new_output_layer"):
            adv_logits = tf.layers.dense(prelogits, n_outputs, name='adv_logits', kernel_initializer=tf.contrib.layers.xavier_initializer())
            y_prob = tf.nn.softmax(adv_logits, name ='y_prob')
            y_pred = tf.reduce_max(adv_logits, axis=1, keepdims=True)

            # y_pred is not differentialble
            # y_pred = tf.one_hot(indices=tf.argmax(y_prob,axis=1), depth=n_outputs)

        # y = tf.placeholder(tf.float32, shape=[None, n_outputs])
        y= tf.placeholder(tf.float32, shape=[None, 1])
        with tf.name_scope('train'):
            # loss function
            # loss = tf.reduce_mean(tf.nn.l2_loss(y_prob - y))
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y , logits=y_pred))
            # tf.summary.scalar('adv_loss', loss)
            optimizer = tf.train.RMSPropOptimizer(0.01)
            adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adv_logits')
            training_op= optimizer.minimize(loss, var_list = adv_vars)


        with tf.name_scope('init_and_save'):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

    model = dict()

    model['training_op'] = training_op
    model['saver'] = saver
    model['X'] = X
    model['y'] = y
    model['training'] = training
    model['y_prob'] = y_prob
    model['loss'] = loss


    #perform init
    num_cpu =1
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.allow_growth = True
    if use_new_model and os.path.exists(model_path_completion('models/adv_model{}.index'.format(use_new_name))):
        model_path = model_path_completion('models/adv_model{}'.format(use_new_name))
    else:
        model_path = model_path_completion('models/inception_v3.ckpt')
    print(colored('init with model from: {}'.format(model_path), 'blue'))

    with tf.device(dev_name):
        # merged = tf.summary.merge_all()
        sess = tf.Session(config = tf_config)

    save_path =log_path_completion('loss_plot/adv'+log_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    with tf.device(dev_name):
        # train_writer = tf.summary.FileWriter(save_path,sess.graph)
        sess.run(init)
        inception_saver.restore(sess, model_path)

    model['sess'] = sess
    # model['merged'] = merged
    # model['train_writer'] = train_writer
    return model



# for new adv policy preparation
def prepare_image2(image, target_width=224, target_height = 224, max_zoom = 0.2):
    random.seed(48)
    height = image.shape[0]
    width = image.shape[1]
    image_ratio= width/height
    target_image_ratio = target_width/target_height
    crop_vertically = image_ratio<target_image_ratio
    crop_width = width if crop_vertically else int(height *target_image_ratio)
    crop_height = int(width/target_image_ratio) if crop_vertically else height

    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height/resize_factor)

    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0+crop_width
    y1= y0+ crop_height

    image = image[y0:y1, x0:x1]

    if np.random.rand()<0.5:
        image = np.fliplr(image)

    image = imresize(image, (target_width, target_height))

    return (image.astype(np.float32)-111)/144


def prepare_X_batch2(image):
    prepared_images = [prepare_image2(image)]
    X_batch = np.stack(prepared_images) 
    print("X_batch: shape", X_batch.shape)
    return X_batch



def prepare_X_batch(image):
    prepared_images = [prepare_image(image)]
    X_batch = 2*np.stack(prepared_images) -1
    print('X_batch:shape', X_batch.shape)
    return X_batch


def prepare_batch(images, labels):
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2*np.stack(prepared_images) -1
    y_batch = labels.astype(np.int32)
    return X_batch, y_batch


def prepare_batch_sample(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :3] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch


def train_adv(model, y_batch, sess, X_batch, total_steps):
    n_epochs = 1
    n_iteration_per_epoch = 1

    # summary = None
    adv_loss = None
    for epoch in range(n_epochs):
        print(colored('Epoch: '.format(epoch), 'cyan'))
        for iteration in range(n_iteration_per_epoch):
            _ , adv_loss = sess.run([model['training_op'], model['loss']], feed_dict={model['X']:X_batch, model['y']:y_batch, model['training']:True})

            print(colored('adv train loss: {}'.format(adv_loss),'cyan'))
    # model['train_writer'].add_summary(summary, total_steps)
    return adv_loss






