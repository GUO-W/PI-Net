# generate bbox_root_mupots_gt for graphU pose refine
#[{"image_id": 0, 
#  "root_cam": [-298.9688720703125, 121.402099609375, 3007.0224609375], 
#  "bbox":     [464.4206237792969, 164.09814453125, 818.8088989257812, 818.8088989257812], 

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

