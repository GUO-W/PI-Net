##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and UPC
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##

import os
import os.path as osp
import sys
import numpy as np

class Config:

    trainset = ['MuCo']

    testset = 'MuPoTS_skeleton'

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '../')
    this_dir = cur_dir.split(osp.sep)[-1]
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir_link = osp.join(cur_dir, 'snapshot')

    ## model setting, input&output
    resnet_type = 50 # 50, 101, 152
    input_shape = (256, 256)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 64
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    # rnn setting
    shuffle_rate=0.75

    ## DATA
    nb_test_seq = 4
    pair_index_path_muco = osp.join(data_dir, trainset[0], 'data','annotations/MuCo_id2pairId.json')
    pair_index_path =  osp.join(data_dir, testset, 'data', 'MuPoTS-3D_id2pairId.json')

    train_annot_path = osp.join(data_dir, trainset[0], 'data','annotations/MuCo-3DHP_with_posenent_result_filter.json')
    val_annot_path = osp.join(data_dir, testset, 'data',  'MuPoTS-3D_with_posenet_result.json') #'MuPoTS-3D.json')

    ## training config
    batch_size = 4#32
    end_epoch = 26#31#21

    lr = 1e-5#5e-5
    lr_strategy = "const"# "poly"#"poly" # "poly"
    # poly
    end_lr = lr * 1e-3
    power = 0.9
    # const
    lr_dec_epoch = [17, 21]
    lr_dec_factor = 10


    ## log
    snapshot_iter = 1
    suffix = this_dir + '_bz' + str(batch_size) + '_lr'+str(lr)
    model_dir = osp.join(output_dir, 'snapshot', suffix)
    vis_dir = osp.join(output_dir, 'vis', suffix)
    tensorboard_log_dir = osp.join(output_dir, 'tensorboard_log', suffix)
    result_dir = osp.join(output_dir, 'result', suffix)
    log_dir = osp.join(output_dir, 'log', suffix)

    ## testing config
    test_batch_size = 1
    flip_test = False#True
    use_gt_info = True#False
    vis_A = False#True

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, bz, lr, snapshot_iter, nb_crossval_split=None, continue_train=False, vis_A=False):
        if gpu_ids:
            self.gpu_ids = gpu_ids
            self.num_gpus = len(self.gpu_ids.split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        self.continue_train = continue_train
        self.vis_A = vis_A
        if bz:
            self.batch_size = bz
        if lr:
            self.lr = lr
        if snapshot_iter:
            self.snapshot_iter = snapshot_iter
        if self.lr_strategy == "poly":
            self.suffix = self.this_dir + '_bz'+str(self.batch_size) + '_polylr'+str(self.lr)
        elif self.lr_strategy == "const":
            self.suffix = self.this_dir + '_bz'+str(self.batch_size) + '_lr'+str(self.lr)
        if nb_crossval_split:
            self.trainset = ['MuPoTS_skeleton']
            self.nb_crossval_split = nb_crossval_split
            self.train_annot_path = osp.join(self.data_dir, self.trainset[0], 'data', 'annotations_crossval', 'mupots_crossval_train' + str(self.nb_crossval_split) + '.json')
            self.val_annot_path = osp.join(self.data_dir, self.testset, 'data', 'annotations_crossval', 'mupots_crossval_val' + str(self.nb_crossval_split) + '.json')
            self.suffix = self.this_dir+'_'+str(self.nb_crossval_split) + '_bz'+str(self.batch_size) + '_lr'+str(self.lr)
        self.model_dir = osp.join(self.output_dir, 'snapshot', self.suffix)
        self.vis_dir = osp.join(self.output_dir, 'vis', self.suffix)
        self.tensorboard_log_dir = osp.join(self.output_dir, 'tensorboard_log', self.suffix)
        self.result_dir = osp.join(self.output_dir, 'result', self.suffix)
        self.log_dir = osp.join(self.output_dir, 'log', self.suffix)

        make_folder(cfg.model_dir)
        make_folder(cfg.log_dir)
        make_folder(cfg.result_dir)
        make_folder(cfg.vis_dir)

        print('>>> Using GPU: {}'.format(self.gpu_ids))
        print ('>>> bz: {}, lr: {}, snapshot: {}'.format(self.batch_size, self.lr, self.snapshot_iter))

cfg = Config()
print ('>>> path:', cfg.cur_dir)
sys.path.insert(0, osp.join(cfg.root_dir))
sys.path.insert(0, osp.join(cfg.root_dir, 'data'))
from utils.dir_utils import add_pypath, make_folder, link_file
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))

