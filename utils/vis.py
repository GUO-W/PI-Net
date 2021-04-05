import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib._png import read_png
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from config import cfg

def vis_2d_keypoints(img_path, nb_person, skeleton, kpts_all): #kpts_all[17*n,2]
    # vis 2d multi person on one img
    kpts_all = np.array(kpts_all).reshape((-1,17,2))

    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    #print(img)
    #img = img.numpy().astype(np.uint8)

    #def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    #kp_mask = np.copy(img)

    for n in range(nb_person):
        kpts = kpts_all[n]
        for l in range(len(skeleton)):
            i1 = skeleton[l][0]
            i2 = skeleton[l][1]
            p1 = kpts[i1,0].astype(np.int32), kpts[i1,1].astype(np.int32)
            p2 = kpts[i2,0].astype(np.int32), kpts[i2,1].astype(np.int32)
            cv2.line(img, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
    return img #cv2.addWeighted(img, 0 , kp_mask, 1, 0)

def vis_result_multiperson_all(gt_all=None, in_all=None, out_all=None, save_path=None):
    # 3d input&output&gt of a img together

    order0 = [0,16,1,2,3,1,5,6,1,15,14,11,12,14,8,9]
    order1 = [16,1,2,3,4,5,6,7,15,14,11,12,13,8,9,10]
    if gt_all is not None:
        nb_person = gt_all.size()[0]
    elif in_all is not None:
        nb_person = in_all.size()[0]
    elif out_all is not None:
        nb_person = out_all.size()[0]
    #print("...nb:", nb_person)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(111, projection='3d')

    for n in range(nb_person):
        if gt_all is not None:
            gt_ = gt_all[n]
            gt_z, gt_x, gt_y = -gt_[:, 1], gt_[:, 0], -gt_[:, 2]
        if in_all is not None:
            in_ = in_all[n]
            in_z, in_x, in_y = -in_[:, 1], in_[:, 0], -in_[:, 2]
        if out_all is not None:
            out_ = out_all[n]
            out_z, out_x, out_y = -out_[:, 1], out_[:, 0], -out_[:, 2]

        for i in range(len(order0)):
            if gt_all is not None:
                ax.plot([gt_x[order0[i]], gt_x[order1[i]]], [gt_y[order0[i]], gt_y[order1[i]]],[gt_z[order0[i]], gt_z[order1[i]]],\
                    c= 'r')
            if in_all is not None:
                ax.plot([in_x[order0[i]], in_x[order1[i]]], [in_y[order0[i]], in_y[order1[i]]],[in_z[order0[i]], in_z[order1[i]]],\
                    c= 'black')
            if out_all is not None:
                ax.plot([out_x[order0[i]], out_x[order1[i]]], [out_y[order0[i]], out_y[order1[i]]],[out_z[order0[i]], out_z[order1[i]]],\
                    c= 'b')

    ax.set_xlabel('x')
    ax.set_xlabel('x')
    ax.set_ylabel('depth')
    ax.set_zlabel('y')
    '''
    if img_path is not None:
        ax.set_zlim(-2.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        arr = read_png(img_path)
        # 10 is equal length of x and y axises of your surface
        stepX, stepY = 100. / arr.shape[1], 100. / arr.shape[0]
        X1 = np.arange(-50,50 , stepX)
        Y1 = np.arange(-50,50, stepY)
        X1, Y1 = np.meshgrid(X1, Y1)
        # stride args allows to determine image quality
        # stride = 1 work slow
        ax.plot_surface(X1, Y1, np.atleast_2d(-2.01), rstride=1, cstride=1, facecolors=arr)
    '''
    plt.savefig(save_path+'_.png')
    print("...one img saved by vis_result_multiperson_all")


def vis_result_multiperson(gt_all, in_all, out_all, save_path):
    # 3d input&output&gt of a img by instance (saved seperately)

    order0 = [0,16,1,2,3,1,5,6,1,15,14,11,12,14,8,9]
    order1 = [16,1,2,3,4,5,6,7,15,14,11,12,13,8,9,10]
    nb_person = gt_all.size()[0]
    #print("...nb:", nb_person)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    for n in range(nb_person):
        gt_ = gt_all[n]
        in_ = in_all[n]
        out_ = out_all[n]
        gt_z, gt_x, gt_y = -gt_[:, 1], gt_[:, 0], -gt_[:, 2]
        in_z, in_x, in_y = -in_[:, 1], in_[:, 0], -in_[:, 2]
        out_z, out_x, out_y = -out_[:, 1], out_[:, 0], -out_[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(order0)):
            ax.plot([gt_x[order0[i]], gt_x[order1[i]]], [gt_y[order0[i]], gt_y[order1[i]]],[gt_z[order0[i]], gt_z[order1[i]]],\
                    c= 'r')
            ax.plot([in_x[order0[i]], in_x[order1[i]]], [in_y[order0[i]], in_y[order1[i]]],[in_z[order0[i]], in_z[order1[i]]],\
                    c= 'b')
            ax.plot([out_x[order0[i]], out_x[order1[i]]], [out_y[order0[i]], out_y[order1[i]]],[out_z[order0[i]], out_z[order1[i]]],\
                    c= 'black')

        ax.set_xlabel('x')
        ax.set_xlabel('x')
        ax.set_ylabel('depth')
        ax.set_zlabel('y')
        plt.savefig(save_path+'_'+str(n)+'.png')
    print("...one img saved by vis_result_multiperson")

def vis_result(pose1, pose2, save_path):

    ##img= np.array(img_patch, dtype=np.float32)
    #gt = np.array(gt, dtype=np.float32).reshape((17, 3))
    p1 = np.array(pose1, dtype=np.float32).reshape((17, 3))
    p2 = np.array(pose2, dtype=np.float32).reshape((17, 3))

    order0 = [0,16,1,2,3,1,5,6,1,15,14,11,12,14,8,9]
    order1 = [16,1,2,3,4,5,6,7,15,14,11,12,13,8,9,10]
    ##limbs = [head,       neck,      R_slder, RUp_arm,   RLow_arm,  L_slder,  LUp_arm,   LLow_arm, spine,  \
            #pelvis,   R_hip,  RUp_leg,  RLow_leg,  L_hip, LUp_leg,    LLow_leg]
	#color =  ['darkgreen','seagreen','black', 'dimgray', 'dimgrey','skyblue','royalblue','navy','darkcyan',\
    #        'darkgreen','gray','darkgray','darkgrey','c',   'dodgerblue','navy']

    p1_z, p1_x, p1_y = -p1[:, 1], p1[:, 0], -p1[:, 2]
    p2_z, p2_x, p2_y = -p2[:, 1], p2[:, 0], -p2[:, 2]

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
    plt.savefig(save_path)#'tmp/'+filename+'_'+name+'.png')

    print("...one img saved by vis_result")


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    #ax.set_xlim([0,cfg.input_shape[1]])
    #ax.set_ylim([0,1])
    #ax.set_zlim([-cfg.input_shape[0],0])
    ax.legend()

    plt.show()
    cv2.waitKey(0)

