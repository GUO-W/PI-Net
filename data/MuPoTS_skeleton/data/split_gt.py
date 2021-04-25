##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and Polytechnic University of Catalonia  [to be checked] (do the other people you collaborate come from this university ?)
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##

import json
import numpy as np
import math
gt = json.load(open('MuPoTS-3D.json','r'))

annotations = gt['annotations']
print("nb of bbox in gt:", len(annotations))

im2labels = dict()
for anno in annotations:
    if anno['image_id'] not in im2labels:
        im2labels[anno['image_id']] = [anno]
    else:
        im2labels[anno['image_id']].append(anno)

thres = math.sqrt(1080**2 + 1920**2)
id2pairId = {}
for anno in annotations:
    idx = anno['id']
    img_id = anno['image_id']
    bbox = anno['bbox']
    center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]

    dis_min  = thres
    id_pair = idx #self
    for cand in im2labels[img_id]:
        bbox_cand = cand['bbox']
        idx_cand = cand['id']
        if (idx_cand != idx):
            center_cand = [bbox_cand[0] + bbox_cand[2]/2, bbox_cand[1] + bbox_cand[3]/2]
            dis = math.sqrt((center[0] - center_cand[0])**2 + (center[1] - center_cand[1])**2)
            if (dis < 0.3 * thres):
                if dis < dis_min:
                    id_pair = idx_cand
                    dis_min = dis

    id2pairId[str(idx)] = id_pair

i = 0
for k in id2pairId:
    if id2pairId[k] == str(k):
        i = i + 1
print("nb of pairs:", len(id2pairId))
print("nb of self pairs:", i)
with open('MuPoTS-3D_id2pairId.json', 'w') as w:
    json.dump(id2pairId, w)
