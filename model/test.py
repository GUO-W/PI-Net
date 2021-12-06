##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and UPC
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##

import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    # keep same with train
    parser.add_argument('--bz', type=int, dest='bz')
    parser.add_argument('--lr', type=float, dest='lr')
    parser.add_argument('--data', type=str, dest='nb_crossval_split')

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, args.bz, args.lr, None, args.nb_crossval_split)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    print(">>> testset:", cfg.val_annot_path)

    if args.test_epoch == 'all':
        # test all epoch (p1) and draw in tensorboardx
        writer = SummaryWriter(cfg.tensorboard_log_dir)
        for epoch in range(cfg.end_epoch):
            tester = Tester(epoch)
            tester._make_batch_generator()
            tester._make_model()

            preds = []
            preds_copain = []

            with torch.no_grad():
                for itr, inp in enumerate(tqdm(tester.batch_generator)):
                    #img_id, img_path, bbox_pair, joint_cam_pair, joint_cam_noise_pair = inp
                    joint_cam_noise_pair,lengths = inp

                    coord_out = tester.model(joint_cam_noise_pair.float().cuda(),lengths.cuda())
                    #coord_out = tester.model(coord_out,lengths.cuda())

                    coord_out = coord_out.cpu().numpy()
                    coord_out_0 = coord_out[:,:17] #1*17*3
                    preds.append(coord_out_0)
            # evaluate
            preds = np.concatenate(preds, axis=0)
            tot_err, tot_err_pa = tester.testset.evaluate_cam(preds)
            writer.add_scalars('scalar/test',{'mpjpe':tot_err, 'PA_mpjpe':tot_err_pa} ,epoch)
        writer.close()
        print("......evaluate done")


    else:
    #test one epoch
        tester = Tester(args.test_epoch)
        tester._make_batch_generator()
        tester._make_model()

        preds = []
        with torch.no_grad():
            for itr, inp in enumerate(tqdm(tester.batch_generator)):
                #img_id, img_path, bbox_pair, joint_cam_pair, joint_cam_noise_pair = inp
                joint_cam_noise_pair,lengths = inp

                # itr
                coord_out =tester.model(joint_cam_noise_pair.float().cuda(),lengths.cuda())
                #coord_out_itr0 = tester.model(joint_cam_noise_pair.float().cuda(),lengths.cuda())
                #coord_out = tester.model(coord_out_itr0,lengths.cuda())

                coord_out = coord_out.cpu().numpy()
                coord_out_0 = coord_out[:,:17] #1*17*3
                preds.append(coord_out_0)
        print("......pred done")

        # evaluate
        preds = np.concatenate(preds, axis=0)
        print("......evaluate")
        tester.testset.evaluate_cam_list(preds,save_mat_result=True)
        print("......evaluate done")

if __name__ == "__main__":
    main()
