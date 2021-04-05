import torch
import numpy as np
from config import cfg
import copy

### 3d transform ###

#def eulerAngles2R(theta):
#def

### coord transfer ###

def trans2cam(pred_2d_kpt, bbox, root, f, c):
    # w,h,z-zp in output space -> x,y,z cam coord
    pred_2d_kpt = warp_coord_to_original(pred_2d_kpt, bbox, root)
    pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)
    return pred_3d_kpt

def pixel2cam_(pixel_coord, f, c):
 	# use in vis_test1img.py
    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    return x,y,z

def pixel2cam(pixel_coord, f, c):
    # w,h,z -> x,y,z
    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return coord

def warp_coord_to_original(joint_out, bbox, center_cam):
    x = joint_out[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
    y = joint_out[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]
    z = (joint_out[:, 2] / cfg.depth_dim * 2. - 1.) * (cfg.bbox_3d_shape[0]/2.) + center_cam[2]
    coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return coord

def trans2cam_bz(pred_2d_kpt, bbox, root, f, c):
    # w,h,z-zp in output space -> x,y,z cam coord
    pred_2d_kpt = warp_coord_to_original_bz(pred_2d_kpt, bbox, root)
    pred_3d_kpt = pixel2cam_bz(pred_2d_kpt, f, c)
    return pred_3d_kpt

def pixel2cam_bz(pixel_coord, f, c):
    x = (pixel_coord[:, :, 0] - c[:, 0:1]) / f[:, 0:1] * pixel_coord[:, :, 2]
    y = (pixel_coord[:, :, 1] - c[:, 1:2]) / f[:, 1:2] * pixel_coord[:, :, 2]
    z = pixel_coord[:, :, 2]
    cam_coord = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)),2)
    #print("cam_coord:",cam_coord)#,cam_coord.float())
    return cam_coord

def warp_coord_to_original_bz(joint_out, bbox, center_cam):
    # joint_out: output from soft-argmax
    x = joint_out[:, :, 0] / cfg.output_shape[1] * bbox[:,2:3] + bbox[:,0:1]
    y = joint_out[:, :, 1] / cfg.output_shape[0] * bbox[:,3:4] + bbox[:,1:2]
    z = (joint_out[:, :, 2] / cfg.depth_dim * 2. - 1.) * (cfg.bbox_3d_shape[0]/2.) + center_cam[:,2:3]
    #print("...1:",x.shape)
    coord = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)),2)
    #print("...2:",coord.shape)
    return coord


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def world2cam(world_coord, R, T):

    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


### rigid alignment ###

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2


### others ###

def get_bbox(joint_img):

    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:,0])
    ymin = np.min(joint_img[:,1])
    xmax = np.max(joint_img[:,0])
    height = ymax - ymin - 1

    bbox[0] = (xmin + xmax)/2. - width/2*1.2
    bbox[1] = (ymin + ymax)/2. - height/2*1.2
    bbox[2] = width*1.2
    bbox[3] = height*1.2

    return bbox


def transform_joint_to_other_db(src_joint, src_name, dst_name):

    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def fliplr_joints(_joints, width, matched_parts):
    """
    flip coords
    joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
    width: image width
    matched_parts: list of pairs
    """
    joints = _joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

