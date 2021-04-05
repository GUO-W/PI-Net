#used in train for skeleton input
import os
import os.path as osp
import numpy as np
import math
from utils.pose_utils import get_bbox
from pycocotools.coco import COCO
from config import cfg
import json
from utils.pose_utils import pixel2cam, get_bbox, warp_coord_to_original, rigid_align, cam2pixel
from utils.vis import vis_keypoints, vis_3d_skeleton
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

class MuCo:
    def __init__(self, data_split, is_val):
        self.data_split = data_split
        #self.img_dir = osp.join('..', 'data', 'MuCo', 'data')
        #self.train_annot_path = osp.join('..', 'data', 'MuCo', 'data', 'MuCo-3DHP.json')
        self.img_dir = osp.join(cfg.data_dir, 'MuCo', 'data')
        self.train_annot_path = cfg.train_annot_path
        self.val_annot_path = cfg.val_annot_path

        self.joint_num = 21
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        self.joints_have_depth = True
        self.root_idx = self.joints_name.index('Pelvis')
        self.is_val = is_val
        self.pair_index_path = cfg.pair_index_path_muco
        self.data = self.load_data()

    def load_data(self):

        if self.data_split == 'train':
            db = COCO(self.train_annot_path)
        #else:
        #    db = COCO(self.val_annot_path)
            #print('Unknown data subset')
            #assert 0

        data = []
        id2pairId = json.load(open(self.pair_index_path,'r'))
        n = 0

        for aid in db.anns.keys():
            #print("aid:",aid)
            ann = db.anns[aid]
            #if ann['is_valid'] == 0:
            #    print("...is valid")
            #    continue

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            fx, fy = img['f']
            cx, cy = img['c']
            f = np.array([fx, fy]); c = np.array([cx, cy]);

            joint_cam = np.array(ann['keypoints_cam'])
            joint_cam_posenet = np.array(ann['keypoints_cam_posenet'])
            root_cam = joint_cam[self.root_idx]

            joint_img = np.array(ann['keypoints_img'])
            joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
            joint_img[:,2] = joint_img[:,2] - root_cam[2]
            joint_vis = np.ones((self.joint_num,1))

            bbox_id = ann['id']
            orig_bbox = ann['bbox']
            bbox = np.array(ann['bbox'])
            img_width, img_height = img['width'], img['height']

			# sanitize bboxes
            #print("...bbox process")
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

            # aspect ratio preserving bbox
            bbox = larger_bbox(bbox)

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
            #print("...132:", [(dis2id[k], k)for k in sorted(dis2id.keys())])
            #n_list = id_list_sorted - bbox_id + n
            for cand_id in id_list_sorted:
                n_list.append(cand_id - bbox_id + n)

            data.append({
                'img_id': image_id,
                'img_path': img_path,
                'id': bbox_id,
                'n_copain': n_copain,
				'n_list': n_list,
                'orig_bbox': orig_bbox,
                'bbox': bbox,
                'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                'joint_vis': joint_vis,
                'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c,
                'joint_cam_posenet': joint_cam_posenet, # result from posenet_nonefine
                #'noise': noise,
            })
            n = n + 1

        return data

    def evaluate(self, preds, result_dir):
        # test for img output, use in test.py
        # add posenet 3d cam result to gt file as 'MuPoTS-3D_with_posenet_result.json', add key 'keypoints_cam_posenet'

        #print('Evaluation start...')
        gts = self.load_data()#self.data
        sample_num = len(preds)
        joint_num = self.joint_num

        pred_2d_per_bbox = {}
        pred_2d_save = {}
        pred_3d_save = {}
        #pred_3d_save_tmp = {}

        gt_dict_orig = json.load(open('/local_scratch/wguo/repos/3DMPPE/posenet_gr/data/MuCo/data/annotations/MuCo-3DHP.json','r'))
        gt_dict = gt_dict_orig

        for n in range(sample_num):
            gt = gts[n]
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            bbox_id = gt['id']
            f = gt['f']
            c = gt['c']
            #if bbox_id != n:
            #    print("...error: bbox_id is not n")

            # restore coordinates to original space
            pred_2d_kpt = preds[n].copy()
            #pred_2d_kpt = np.take(pred_2d_kpt, self.eval_joint, axis=0)
            pred_2d_kpt = warp_coord_to_original(pred_2d_kpt, bbox, gt_3d_root)

            if str(n) in pred_2d_per_bbox:
                pred_2d_per_bbox[str(n)].append(pred_2d_kpt)
            else:
                pred_2d_per_bbox[str(n)] = [pred_2d_kpt]

            pred_2d_kpt = pred_2d_per_bbox[str(n)].copy()
            pred_2d_kpt = np.mean(np.array(pred_2d_kpt), axis=0)
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

			### add posenet 3d cam result to gt file as 'MuCo_with_posenet_result.json', add key 'keypoints_cam_posenet'
            gt_dict['annotations'][int(bbox_id)]['keypoints_cam_posenet'] = pred_3d_kpt.tolist()

        # filter.py
        #for anno in gt_dict['annotations']:
        #    if 'keypoints_cam_posenet' not in anno.keys():
        #        del anno

        with open('/local_scratch/wguo/repos/3DMPPE/posenet_gr/data/MuCo/MuCo_with_posenet_result.json','w') as w:
            json.dump(gt_dict, w)



