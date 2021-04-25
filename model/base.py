# base.py is derived from [3DMPPE_POSENET_RELEASE](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git)
# distributed under MIT License (c) 2019 Gyeongsik Moon.


import os
import os.path as osp
import math
import time
import glob
import abc
import random

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from dataset import DatasetLoader
from utils.timer import Timer
from utils.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
#from network import get_model
from model import get_model
import datetime

class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, shuffle_rate=0):
        self.shuffle_rate = shuffle_rate

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label) tuples
        return:
            xs - a tensor of all examples in the batch after padding
            ys - a tensor of all labels in batch
        """
        input_tensors, joint_cam_tensors, joint_vis_tensors, joints_have_depth_tensors, lengths =[], [], [], [], []
        for i, inp in enumerate(batch):
            if len(inp) == 4:
                input_pair, joint_cam_pair, joint_vis_pair, joints_have_depth = inp #train #inpt, targt = inp
                input_pair = torch.Tensor(input_pair)
                input_tensors.append(input_pair)
                joint_cam_tensors.append(torch.Tensor(joint_cam_pair))
                joint_vis_tensors.append(torch.Tensor(joint_vis_pair))
                joints_have_depth_tensors.append(torch.Tensor(joints_have_depth))
                lengths.append(input_pair.size(0)/17)
            else:
                input_pair = inp  #test
                input_pair = torch.Tensor(input_pair)
                input_tensors.append(input_pair)
                lengths.append(input_pair.size(0)/17)

        lengths = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
        padded_input = pad_sequence(input_tensors, batch_first=True)
        if len(joint_cam_tensors) != 0:
            padded_joint_cam = pad_sequence(joint_cam_tensors, batch_first=True)#, padding_value=-1)
            padded_joint_vis = pad_sequence(joint_vis_tensors, batch_first=True)#, padding_value=-1)
            padded_joint_have_depth = pad_sequence(joints_have_depth_tensors, batch_first=True)#, padding_value=-1)
            return padded_input, padded_joint_cam, padded_joint_vis,padded_joint_have_depth, lengths
        else:
            return padded_input,lengths



    def shuffle_batch(self, batch):
        """
        Permute the bboxes around, batch is a list of (input_tensor, target_tensor) tuples
        """
        for i, inp in enumerate(batch):
            input_pair, gt_pair, joint_vis_pair = inp
            seq_len = inpt.size(0)
            order = np.random.permutation(seq_len)
            batch[i] = (input[order, :], targt[order])

    def __call__(self, batch):
        return self.pad_collate(batch)

t = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
# dynamic dataset import
for i in range(len(cfg.trainset)):
    exec('from ' + cfg.trainset[i] + ' import ' + cfg.trainset[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name= t + '_logs.txt'):

        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar'))
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])

        return start_epoch, model, optimizer

class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__(log_name = t + '_train_logs.txt')
    def get_optimizer(self, model):

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        if cfg.lr_strategy == "poly":
            for g in self.optimizer.param_groups:
                g['lr'] = (cfg.lr - cfg.end_lr) * (1 - epoch / cfg.end_epoch) ** cfg.power + cfg.end_lr
        if cfg.lr_strategy == "const":
            for e in cfg.lr_dec_epoch:
                if epoch < e:
                    break
            if epoch < cfg.lr_dec_epoch[-1]:
                idx = cfg.lr_dec_epoch.index(e)
                for g in self.optimizer.param_groups:
                    g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
            else:
                for g in self.optimizer.param_groups:
                    g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr
    def _make_batch_generator(self):
        # data load and construct batch generator
        #self.logger.info("Creating dataset...")
        trainset_loader = []
        batch_generator = []
        iterator = []
        valset_loader = []
        val_batch_generator = []
        val_iterator = []
        for i in range(len(cfg.trainset)):
            if i > 0:
                ref_joints_name = trainset_loader[0].joints_name
            else:
                ref_joints_name = None
            # train
            trainset_loader.append(DatasetLoader(eval(cfg.trainset[i])("train",False), ref_joints_name, True, \
                transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])))

            batch_generator.append(DataLoader(dataset=trainset_loader[-1],\
                                    batch_size=cfg.num_gpus*cfg.batch_size//len(cfg.trainset), \
                                    shuffle=True,\
                                    collate_fn = PadCollate(cfg.shuffle_rate),\
                                    num_workers=cfg.num_thread,\
                                    pin_memory=True))

            iterator.append(iter(batch_generator[-1]))


            # val
            valset_loader.append(DatasetLoader(eval(cfg.testset)("train",True), ref_joints_name, True,\
                    transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])))

            val_batch_generator.append(DataLoader(dataset=valset_loader[-1],\
                                        batch_size=cfg.num_gpus*cfg.batch_size//len(cfg.trainset),\
                                        shuffle=True,\
                                        collate_fn = PadCollate(),\
                                        num_workers=cfg.num_thread, \
                                        pin_memory=True))

            val_iterator.append(iter(val_batch_generator[-1]))



        self.joint_num = trainset_loader[0].joint_num
        self.itr_per_epoch = math.ceil(trainset_loader[0].__len__() / cfg.num_gpus / (cfg.batch_size // len(cfg.trainset)))
        self.batch_generator = batch_generator
        self.iterator = iterator

        self.val_itr_per_epoch = math.ceil(valset_loader[0].__len__() / cfg.num_gpus / (cfg.batch_size // len(cfg.trainset)))
        self.val_batch_generator = val_batch_generator
        self.val_iterator = val_iterator


    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(pretrained = True).cuda() #get_pose_net(cfg, True, self.joint_num)
        #model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = t + '_test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        #self.logger.info("Creating dataset...")
        testset = eval(cfg.testset)("test",True) #is_val = True

        testset_loader = DatasetLoader(testset, None, False,\
                transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]))

        if cfg.vis_A:
            batch_generator = DataLoader(dataset=testset_loader,\
                    batch_size=cfg.num_gpus*cfg.test_batch_size,\
                    num_workers=cfg.num_thread,\
                    pin_memory=True)
        else:
            batch_generator = DataLoader(dataset=testset_loader,\
                    batch_size=cfg.num_gpus*cfg.test_batch_size,\
                    shuffle=False,\
                    collate_fn = PadCollate(),\
                    num_workers=cfg.num_thread,\
                    pin_memory=True)

        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.batch_generator = batch_generator

    def _make_model(self):

        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # load model
        self.logger.info("Creating graph...")
        model = get_model(pretrained = False).cuda()#get_pose_net(cfg, False, self.joint_num)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])

        model.eval()

        self.model = model

    def _evaluate(self, preds,preds_copain, result_save_path=None):
        self.testset.evaluate_cam(preds, preds_copain)#, result_save_path)

