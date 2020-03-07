from adv_policy import build_network
from adv_policy import prepare_batch_sample
from adv_policy import train_adv

import sys
import tarfile
from six.moves import urllib
import os
from collections import defaultdict
import numpy as np


FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("datasets", "flowers")

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()



def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)




if __name__ =='__main__':

    if not os.path.exists(FLOWERS_PATH):
        fetch_flowers()

    flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")
    flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                             if os.path.isdir(os.path.join(flowers_root_path, dirname))])

    image_paths = defaultdict(list)

    for flower_class in flower_classes:
        image_dir = os.path.join(flowers_root_path, flower_class)
        for filepath in os.listdir(image_dir):
            if filepath.endswith(".jpg"):
                image_paths[flower_class].append(os.path.join(image_dir, filepath))

    flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
    flower_paths_and_classes = []
    for flower_class, paths in image_paths.items():
        for path in paths:
            flower_paths_and_classes.append((path, flower_class_ids[flower_class]))

    test_ratio = 0.2
    train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))

    np.random.shuffle(flower_paths_and_classes)

    flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
    flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

    X_batch, y_batch = prepare_batch_sample(flower_paths_and_classes_train, batch_size=4)

    model = build_network()

    model_path = 'models/inception_v3.ckpt'
    train_adv(model, model_path, X_batch, y_batch)


