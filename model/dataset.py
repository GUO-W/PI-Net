##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and Polytechnic University of Catalonia  [to be checked] (do the other people you collaborate come from this university ?)
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##

import numpy as np
import cv2
import random
import time
import torch
import copy
import math
from torch.utils.data.dataset import Dataset
from utils.vis import vis_keypoints, vis_3d_skeleton, vis_result
from utils.pose_utils import fliplr_joints, transform_joint_to_other_db, trans2cam#, eulerAngles2R
from config import cfg
import json
import matplotlib.pyplot as plt

class DatasetLoader(Dataset):
    def __init__(self, db, ref_joints_name, is_train, transform):

        self.db = db.data
        self.joint_num = db.joint_num
        self.skeleton = db.skeleton
        self.flip_pairs = db.flip_pairs
        self.joints_have_depth = db.joints_have_depth
        self.joints_name = db.joints_name
        self.ref_joints_name = ref_joints_name

        self.transform = transform
        self.is_train = is_train
        self.is_val = db.is_val
        if self.is_train and not self.is_val:
            self.do_augment = True
        else:
            self.do_augment = False

        self.pair_index_path = db.pair_index_path

    def __getimg__(self, index):

        joint_num = self.joint_num
        skeleton = self.skeleton
        flip_pairs = self.flip_pairs
        joints_have_depth = self.joints_have_depth

        data = copy.deepcopy(self.db[index])

        img_id = data['img_id']
        img_path = data['img_path']

        bbox = data['bbox']
        gt_3D_root = data['root_cam']
        f = data['f']
        c = data['c']
        id_list = data['n_list'] # instances in same img (from near to far) , including itself

        joint_img = data['joint_img']
        joint_cam = data['joint_cam']
        joint_vis = data['joint_vis']

        joint_cam_posenet = data['joint_cam_posenet']

        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape

        scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

        img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion)
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

        if do_flip:
            joint_img[:, 0] = img_width - joint_img[:, 0] - 1
            for pair in flip_pairs:
                joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :], joint_img[pair[0], :].copy()
                joint_vis[pair[0], :], joint_vis[pair[1], :] = joint_vis[pair[1], :], joint_vis[pair[0], :].copy()

        for i in range(len(joint_img)):
            joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
            joint_img[i, 2] /= (cfg.bbox_3d_shape[0]/2.) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
            joint_img[i, 2] = (joint_img[i,2] + 1.0)/2. # 0~1 normalize
            joint_vis[i] *= (
                            (joint_img[i,0] >= 0) & (joint_img[i,0] < cfg.input_shape[1]) & \
                            (joint_img[i,1] >= 0) & (joint_img[i,1] < cfg.input_shape[0]) & \
                            (joint_img[i,2] >= 0) & (joint_img[i,2] < 1)
                            )


        # change coordinates to output space
        joint_img[:, 0] = joint_img[:, 0] / cfg.input_shape[1] * cfg.output_shape[1]
        joint_img[:, 1] = joint_img[:, 1] / cfg.input_shape[0] * cfg.output_shape[0]
        joint_img[:, 2] = joint_img[:, 2] * cfg.depth_dim

        if self.is_train:
            img_patch = self.transform(img_patch)

            if self.ref_joints_name is not None:
                joint_img = transform_joint_to_other_db(joint_img, self.joints_name, self.ref_joints_name)
                joint_cam = transform_joint_to_other_db(joint_cam, self.joints_name, self.ref_joints_name)
                joint_cam_posenet = transform_joint_to_other_db(joint_cam_posenet, self.joints_name, self.ref_joints_name)
                joint_vis = transform_joint_to_other_db(joint_vis, self.joints_name, self.ref_joints_name)

            joint_img = joint_img.astype(np.float32)
            joint_cam = joint_cam.astype(np.float32)
            joint_cam_posenet = joint_cam_posenet.astype(np.float32)
            joint_vis = (joint_vis > 0).astype(np.float32)
            joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

            return id_list,img_patch, bbox, gt_3D_root, f, c, joint_img, joint_cam, joint_cam_posenet, joint_vis, joints_have_depth, None#noise
        else:
            img_patch = self.transform(img_patch)
            return img_id, img_path, id_list, img_patch, bbox, gt_3D_root, f, c, joint_img, joint_cam, joint_cam_posenet, None# noise

    def __getitem__(self, index):
        if self.is_train:
            id_list, _, _, _, _, _, _, joint_cam, joint_cam_posenet, joint_vis, joints_have_depth, _ = self.__getimg__(index)
            if index != id_list[0]:
                print("...ERROR!! dataset.py lin134, IN generating id_list in the img")
            if len(id_list)==1:
                id_list = [index,index]

            joint_cam_posenet_pair, joint_cam_pair, joint_vis_pair = joint_cam_posenet[:17,], joint_cam[:17,], joint_vis[:17,]
            for pair_index in id_list[1:]:
                _, _, _, _, _, _, _, joint_cam_m, joint_cam_posenet_m, joint_vis_m, _, _ = self.__getimg__(pair_index)
                joint_cam_posenet_pair = np.concatenate((joint_cam_posenet_pair, joint_cam_posenet_m[:17,]), axis=0)
                joint_cam_pair = np.concatenate((joint_cam_pair, joint_cam_m[:17,]), axis=0)
                joint_vis_pair = np.concatenate((joint_vis_pair, joint_vis_m[:17,]), axis=0) # 34*1 -> (17*N)*1 , N = nb of instances in img - 1

            vis = False#True
            if vis:
                vis_patch_data(img_patch, joint_img_noise_1, joint_img, joint_img_m)

            return joint_cam_posenet_pair,\
                   joint_cam_pair, joint_vis_pair, joints_have_depth

        else:
            img_id, img_path, id_list, _, bbox, _, _, _, _, joint_cam, joint_cam_posenet, _  = self.__getimg__(index)
            if index != id_list[0]:
                print("...ERROR!! dataset.py lin156, IN generating id_list in the img")
            if len(id_list)==1:
                print("...ERROR!! dataset.py lin158, IN generating id_list in the img:only one instance exits in certain img")

            bbox_pair, joint_cam_pair, joint_cam_posenet_pair = np.expand_dims(bbox,axis=0), joint_cam, joint_cam_posenet
            for pair_index in id_list[1:]:
                _, _, _, _, bbox_m, _, _, _, _, joint_cam_m, joint_cam_posenet_m, _  = self.__getimg__(pair_index)
                bbox_pair = np.concatenate((bbox_pair, np.expand_dims(bbox_m,axis=0)), axis=0)
                joint_cam_pair = np.concatenate((joint_cam_pair, joint_cam_m), axis=0)
                joint_cam_posenet_pair = np.concatenate((joint_cam_posenet_pair, joint_cam_posenet_m), axis=0)

            if cfg.vis_A:
                return img_id, img_path, bbox_pair, joint_cam_pair, joint_cam_posenet_pair
            else:
                return joint_cam_posenet_pair


    def __len__(self):
        return len(self.db)

    def mirror_gt(self, joint, vis = None):
        if vis is None:
            joint_img = joint.copy()
            joint_img[:, 0] = cfg.output_shape[0] - joint_img[:, 0] - 1
            for pair in self.flip_pairs:
                tmp = joint_img[pair[0], :].copy()
                joint_img[pair[0], :] = joint_img[pair[1], :]
                joint_img[pair[1], :] = tmp
            return joint_img
        else:
            joint_img = joint.copy()
            joint_vis = vis.copy()

            joint_img[:, 0] = cfg.output_shape[0] - joint_img[:, 0] - 1
            for pair in self.flip_pairs:
                tmp = joint_img[pair[0], :].copy()
                joint_img[pair[0], :] = joint_img[pair[1], :]
                joint_img[pair[1], :] = tmp

                v = joint_vis[pair[0], :].copy()
                joint_vis[pair[0], :] = joint_vis[pair[1], :]
                joint_vis[pair[1], :] = v
            return joint_img, joint_vis

    def handshake_gt(self, joint, bbox, vis = None):
        offset = joint[4] - joint[7]
        joint_new = joint.copy()
        for i in range(len(joint)):
            joint_new[i] = joint[i] + offset

        bbox_new = [ bbox[0]+offset[0], bbox[1]+offset[1], bbox[2], bbox[3] ]

        if vis is None:
            return joint_new, bbox_new
        else:
            return joint_new, bbox_new, vis


# helper functions
def add_random_noise(A,noise):
    # noise = [mask,R,t,shift], A: joint17*3
    # R 0~10 degree; t 0~1 ; shift 0~1

    # R and T: all/some joints around the root(joint[14])
    #theta = noise[1]*2 #np.random.rand(3)*(10/180 * np.pi) #~10
    #R = eulerAngles2R(theta)
    #t = noise[2]  #np.random.rand(3)
    #A2 = np.transpose(np.dot(R, np.transpose(A))) + t

    # shift
    #keypoint_mask = noise[0]#np.random.randint(2, size=(17,1))
    #keypoint_mask = np.repeat(keypoint_mask, 3, axis=-1)
    #shift = noise[3] * keypoint_mask #np.random.rand(17,3) * keypoint_mask
    shift = np.array(noise[3]) * 100
    A2 = A + shift
    return A2 #joint_noise

def vis_patch_data(img_patch, gt, pose1, pose2):
    img= np.array(img_patch, dtype=np.float32)
    gt = np.array(gt, dtype=np.float32).reshape((17, 3))
    p1 = np.array(pose1, dtype=np.float32).reshape((17, 3))
    p2 = np.array(pose2, dtype=np.float32).reshape((17, 3))

    order0 = [0,16,1,2,3,1,5,6,1,15,14,11,12,14,8,9]
    order1 = [16,1,2,3,4,5,6,7,15,14,11,12,13,8,9,10]
    color =  ['darkgreen','seagreen','black', 'dimgray', 'dimgrey','skyblue','royalblue','navy','darkcyan',\
            'darkgreen','gray','darkgray','darkgrey','c',   'dodgerblue','navy']

    gt_z, gt_x, gt_y = -gt[:, 1], gt[:, 0], -gt[:, 2]
    p1_z, p1_x, p1_y = -p1[:, 1], p1[:, 0], -p1[:, 2]
    p2_z, p2_x, p2_y = -p2[:, 1], p2[:, 0], -p2[:, 2]

    filename = str(random.randrange(1,500))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(order0)):
        ax.plot([gt_x[order0[i]], gt_x[order1[i]]], [gt_y[order0[i]], gt_y[order1[i]]],[gt_z[order0[i]], gt_z[order1[i]]],\
                c= color[i])#'b')
        ax.plot([p1_x[order0[i]], p1_x[order1[i]]], [p1_y[order0[i]], p1_y[order1[i]]],[p1_z[order0[i]], p1_z[order1[i]]],\
                c= 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('depth')
    ax.set_zlabel('y')
    plt.savefig('tmp/'+filename+'gt_p1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(order0)):
        ax.plot([p1_x[order0[i]], p1_x[order1[i]]], [p1_y[order0[i]], p1_y[order1[i]]],[p1_z[order0[i]], p1_z[order1[i]]],\
                c= 'r')
        ax.plot([p2_x[order0[i]], p2_x[order1[i]]], [p2_y[order0[i]], p2_y[order1[i]]],[p2_z[order0[i]], p2_z[order1[i]]],\
                c= 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('depth')
    ax.set_zlabel('y')
    plt.savefig('tmp/'+filename+'p1_p2.png')

    print("...one img saved for vis_patch_data")

def get_aug_config():

    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    do_occlusion = random.random() <= 0.5

    return scale, rot, do_flip, color_scale, do_occlusion


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    # synthetic occlusion
    if do_occlusion:
        while True:
            area_min = 0.0
            area_max = 0.7
            synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]

            ratio_min = 0.3
            ratio_max = 1/0.3
            synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

            synth_h = math.sqrt(synth_area * synth_ratio)
            synth_w = math.sqrt(synth_area / synth_ratio)
            synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
            synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]

            if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
                xmin = int(synth_xmin)
                ymin = int(synth_ymin)
                w = int(synth_w)
                h = int(synth_h)
                img[ymin:ymin+h, xmin:xmin+w, :] = np.random.rand(h, w, 3) * 255
                break

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

