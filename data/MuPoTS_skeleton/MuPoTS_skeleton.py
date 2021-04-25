##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and Polytechnic University of Catalonia  [to be checked] (do the other people you collaborate come from this university ?)
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##

import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
from config import cfg
import json
import cv2
import random
import math
from utils.pose_utils import pixel2cam, get_bbox, warp_coord_to_original, rigid_align, cam2pixel, trans2cam
from utils.vis import vis_keypoints, vis_3d_skeleton
import datetime
import cv2 as cv

def larger_bbox(bbox):
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox


class MuPoTS_skeleton:
    def __init__(self, data_split, is_val):
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'MuPoTS_skeleton', 'data', 'MultiPersonTestSet')

        self.train_annot_path = cfg.train_annot_path
        self.val_annot_path = cfg.val_annot_path
        self.nb_test_seq = cfg.nb_test_seq

        self.human_bbox_root_dir = osp.join('..', 'data', 'MuPoTS_skeleton', 'bbox_root', 'bbox_root_mupots_output.json')#'bbox_root_mupots_gt.json')
        self.joint_num = 21 # MuCo-3DHP
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe') # MuCo-3DHP
        self.original_joint_num = 17 # MuPoTS
        self.original_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head') # MuPoTS
        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7) )
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.joints_have_depth = True
        self.root_idx = self.joints_name.index('Pelvis')
        self.is_val = is_val
        self.pair_index_path = cfg.pair_index_path
        self.data = self.load_data()

    def load_data(self):

        if self.is_val:
            db = COCO(self.val_annot_path)
        else:
            db = COCO(self.train_annot_path)

        id2pairId = json.load(open(self.pair_index_path,'r'))

        data = []
        # use gt bbox and root
        if cfg.use_gt_info: #false
            print("Get bounding box and root from groundtruth")
            n = 0
            for aid in db.anns.keys():
                ann = db.anns[aid]

                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);

                joint_cam = np.array(ann['keypoints_cam'])
                joint_cam_posenet = np.array(ann['keypoints_cam_posenet'])
                root_cam = joint_cam[self.root_idx]

                joint_img = np.array(ann['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_img[:,2] = joint_img[:,2] - root_cam[2]
                joint_vis = np.ones((self.original_joint_num,1))

                bbox_id = ann['id']
                orig_bbox = ann['bbox']
                bbox = np.array(ann['bbox'])
                img_width, img_height = img['width'], img['height']

                x, y, w, h = bbox
                center = [x+w/2, y+h/2]
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
                if w*h > 0 and x2 >= x1 and y2 >= y1:
                    bbox = np.array([x1, y1, x2-x1, y2-y1])
                else:
                    print("sanitize bboxes:",image_id)
                    continue

                #bbox = larger_bbox(bbox)

                n_copain = id2pairId[str(bbox_id)] - bbox_id + n # n_copain - n = id_copain - id

                id_list = db.getAnnIds(image_id) # ids of instances in same img
                dis2id = {}
                n_list = []
                for cand_id in id_list:
                    bbox_cand = db.loadAnns(cand_id)[0]['bbox']
                    center_cand = [bbox_cand[0] + bbox_cand[2]/2, bbox_cand[1] + bbox_cand[3]/2]
                    dis = math.sqrt((center[0] - center_cand[0])**2 + (center[1] - center_cand[1])**2)
                    dis2id[dis] = cand_id
                id_list_sorted = [dis2id[k] for k in sorted(dis2id.keys())]
                for cand_id in id_list_sorted:
                    n_list.append(cand_id - bbox_id + n)

                data.append({
                    'img_id': image_id,
                    'img_path': img_path,
                    'id': bbox_id,
                    'n_copain': n_copain, # n of pair instance (the most close one)
                    'n_list': n_list, # n of instances in same img (from near to far)
                    'orig_bbox': orig_bbox,
                    'bbox': bbox,
                    'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'joint_vis': joint_vis,
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c,
                    'joint_cam_posenet': joint_cam_posenet,
                })
                n = n + 1

        else:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            print("...annot len:",len(annot))#20899 nb of bbox
            for i in range(len(annot)):
                image_id = annot[i]['image_id']
                img = db.loadImgs(image_id)[0]
                img_width, img_height = img['width'], img['height']
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);
                root_cam = np.array(annot[i]['root_cam']).reshape(3)
                bbox = np.array(annot[i]['bbox']).reshape(4)

                data.append({
                    'img_id': image_id,
                    'img_path': img_path,
                    'bbox': bbox,
                    'joint_img': np.zeros((self.original_joint_num, 3)), # dummy
                    'joint_cam': np.zeros((self.original_joint_num, 3)), # dummy
                    'joint_vis': np.zeros((self.original_joint_num, 1)), # dummy
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c,
                })

        return data

    def evaluate_cam_list(self, preds, preds_list=None, save_mat_result=False):
        # test for cam output
        # used when input_img (nb_person not certain, not a pair)
        # used in test.py(save_mat_result=True)

        gts = self.load_data()#self.data
        sample_num = len(preds)
        joint_num = self.original_joint_num

        pred_3d_per_bbox = {}
        pred_2d_save = {}
        pred_3d_save = {}

        for n in range(sample_num):
            gt = gts[n]
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            bbox_id = gt['id']

            pred_3d_kpt = preds[n].copy()

            if str(n) in pred_3d_per_bbox:
                pred_3d_per_bbox[str(n)].append(pred_3d_kpt)
            else:
                pred_3d_per_bbox[str(n)] = [pred_3d_kpt]

                    error = np.zeros((sample_num, self.original_joint_num)) # joint error
        error_pa = np.zeros((sample_num, self.original_joint_num))
        error_seq = {}#[ [] for _ in range(self.nb_test_seq) ] # error for each sequence
        error_seq_pa = {}#[ [] for _ in range(self.nb_test_seq) ]
        for n in range(sample_num):
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']
            gt_vis = gt['joint_vis']
            bbox_id = gt['id']
            img_name = gt['img_path'].split('/')
            img_name = img_name[-2] + '_' + img_name[-1].split('.')[0] # e.g., TS1_img_0001 orig gt 'file_name':TS1/img_0001.jpg

            pred_3d_kpt = pred_3d_per_bbox[str(n)].copy()
            pred_3d_kpt = np.mean(np.array(pred_3d_kpt), axis=0)#np.mean(pred_2d_kpt)

            pred_2d_kpt = cam2pixel(pred_3d_kpt, f, c)

            # save pred result
            if img_name in pred_2d_save:
                pred_2d_save[img_name].append(pred_2d_kpt[:,:2])
                pred_3d_save[img_name].append(pred_3d_kpt)
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:,:2]]
                pred_3d_save[img_name] = [pred_3d_kpt]

            ### calculate average error MPJPE & PA-MPJPE
            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[self.root_idx]
            # rigid alignment for PA MPJPE (protocol #1)
            pred_3d_kpt_pa = rigid_align(pred_3d_kpt, gt_3d_kpt)
            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2,1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2,1))
            seq_idx = int(img_name[2:img_name.find('_')])
            if str(seq_idx) in error_seq:
                error_seq[str(seq_idx)].append(error[n].copy())
                error_seq_pa[str(seq_idx)].append(error_pa[n].copy())
            else:
                error_seq[str(seq_idx)] = [error[n].copy()]
                error_seq_pa[str(seq_idx)] = [error_pa[n].copy()]
        # PAMPJPE
        tot_err_pa = np.mean(error_pa)
        eval_summary = 'Protocol 1' + ' error ( PA MPJPE) >> tot: %.2f\n' % (tot_err_pa)
        for k in error_seq_pa:
            err_pa = np.mean(np.array(error_seq_pa[k]))
            eval_summary += ('TS' + k + ': %.2f ' % err_pa)
        # MPJPE
        tot_err = np.mean(error)
        eval_summary += '\nProtocol 2' + ' error ( MPJPE) >> tot: %.2f\n' % (tot_err)
        for k in error_seq:
            err = np.mean(np.array(error_seq[k]))
            eval_summary += ('TS' + k + ': %.2f ' % err)

        print(eval_summary)

        if not save_mat_result:
            return tot_err, tot_err_pa
        else:
            ### save result in result_mpjpe_sequence.txt
            result = 'pa mpjpe:\n'
            for i in range(20):
                ts = str(i+1)
                error = np.mean(np.array(error_seq_pa[ts]))
                result += (' %.2f \n' %error)
            result = 'mpjpe:\n'
            for i in range(20):
                ts = str(i+1)
                error = np.mean(np.array(error_seq[ts]))
                result += (' %.2f \n' %error)
            txt_path = osp.join(cfg.cur_dir, 'result_mpjpe_sequence.txt')
            with open(txt_path, 'w') as w:
                w.write(''.join(result))

            ### prediction save for calculating pck
            t = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            output_path = osp.join(cfg.result_dir, t+ '_preds_2d_kpt_mupots.mat')
            sio.savemat(output_path, pred_2d_save)
            print("Testing result is saved at " + output_path)
            output_path = osp.join(cfg.result_dir, t+ '_preds_3d_kpt_mupots.mat')
            sio.savemat(output_path, pred_3d_save)
            print("Testing result is saved at " + output_path)



