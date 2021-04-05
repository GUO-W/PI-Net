import json
import os

data = json.load(open('MuCo-3DHP_with_posenet_result.json','r'))
imgs = data['images']
annos = data['annotations']

# 'image_id' : id
im2labels = dict()
for anno in annos:
    if anno['image_id'] not in im2labels:
        im2labels[anno['image_id']] = [anno]
    else:
        im2labels[anno['image_id']].append(anno)


data_filter = {}
data_filter['images'] = []
data_filter['annotations'] = []

for img in imgs:
    img_path = '../' + img['file_name']
    if os.path.exists(img_path):
        data_filter['images'].append(img)
        for ann in im2labels[img['id']]:
            if 'keypoints_cam_posenet' in ann.keys():
                data_filter['annotations'].append(ann)
        #data_filter['annotations'] += im2labels[img['id']]

print('ann len:', len(data_filter['annotations']))
print('img len:', len(data_filter['images']))

#for anno in data_filter['annotations']:
#    if 'keypoints_cam_posenet' not in anno.keys():
#        data_filter['annotations'].remove(anno)

#print('ann len:', len(data_filter['annotations']))
#print('img len:', len(data_filter['images']))

with open('MuCo-3DHP_with_posenent_result_filter.json','w') as w:
    json.dump(data_filter, w)
