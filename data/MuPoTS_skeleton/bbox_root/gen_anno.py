##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and UPC
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##
import json

with open("../data/MuPoTS-3D.json",'r') as f1:
    data_dict = json.load(f1)
    print("data keys:", data_dict.keys())


bbox_root_list = []

for i in range(len(data_dict['annotations'])):
    bbox_root_dict = {}
    bbox_root_dict['image_id'] = data_dict['annotations'][i]['image_id']
    bbox_root_dict['bbox'] = data_dict['annotations'][i]['bbox']
    bbox_root_dict['root_cam'] = data_dict['annotations'][i]['keypoints_cam'][14] #pelvis
    bbox_root_list.append(bbox_root_dict)


with open("bbox_root_mupots_gt.json",'w') as f2:
    json.dump(bbox_root_list, f2)

