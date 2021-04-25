# dir_utils.py is derived from [3DMPPE_POSENET_RELEASE](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git)
# distributed under MIT License (c) 2019 Gyeongsik Moon.

import os
import sys

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
        os.system('ln -s {} {}'.format(src, target))
