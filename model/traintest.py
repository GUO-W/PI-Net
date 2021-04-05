import argparse
from config import cfg
import torch
from base import Trainer, Tester
import torch.backends.cudnn as cudnn
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
from utils.dir_utils import link_file
from model import get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--bz', type=int, dest='bz')
    parser.add_argument('--lr', type=float, dest='lr')
    parser.add_argument('--data', type=str, dest='nb_crossval_split')
    # train
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--snap', type=int, dest='snapshot_iter')
    #test
    parser.add_argument('--test',dest='test', action='store_true')
    #parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    # argument parse and create log
    setup_seed(20)
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.bz, args.lr, args.snapshot_iter, args.nb_crossval_split, args.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True

    print(">>> trainset:", cfg.train_annot_path)
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    #link_file(cfg.model_dir, cfg.model_dir_link)
    writer = SummaryWriter(cfg.tensorboard_log_dir)

    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        ### 1.train ###
        train_loss_per_epoch, train_loss_x, train_loss_y, train_loss_z = 0.0, 0.0, 0.0, 0.0
        pbar_train = tqdm(range(trainer.itr_per_epoch), file=sys.stdout,bar_format=bar_format)
        for itr in pbar_train:
            try:
                joint_cam_noise_pair, joint_cam_pair, joint_vis_pair, joints_have_depth, lengths= next(trainer.iterator[0])
            except StopIteration:
                trainer.iterator[0] = iter(trainer.batch_generator[0])
                joint_cam_noise_pair, joint_cam_pair, joint_vis_pair, joints_have_depth, lengths= next(trainer.iterator[0])
            target = {'coord_cam': joint_cam_pair.cuda(), 'vis': joint_vis_pair.cuda(), 'have_depth': joints_have_depth.cuda()}
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()

            ### forward
            loss_coord , loss_x, loss_y, loss_z = trainer.model(joint_cam_noise_pair.cuda(),  lengths.cuda(), target)
            loss_coord = loss_coord.mean()
            loss_x, loss_y, loss_z = loss_x.mean(), loss_y.mean(), loss_z.mean()

            ### backward
            loss = loss_coord
            loss.backward()
            trainer.optimizer.step()

            print_str = 'Epoch{}/{}'.format(epoch, cfg.end_epoch) \
                     + ' Iter{}/{}:'.format(itr, trainer.itr_per_epoch) \
                     + ' lr=%g' % (trainer.get_lr()) \
                     + ' loss_coord=%.4f' % (loss_coord.detach())\
                     + ' loss_x=%.2f' % (loss_x.detach())\
                     + ' loss_y=%.2f' % (loss_y.detach())\
                     + ' loss_z=%.2f' % (loss_z.detach())
            pbar_train.set_description(print_str, refresh=False)

            train_loss_per_epoch += loss.item()
            train_loss_x += loss_x.item()
            train_loss_y += loss_y.item()
            train_loss_z += loss_z.item()

        train_loss_per_epoch = train_loss_per_epoch/trainer.itr_per_epoch
        train_loss_x = train_loss_x/trainer.itr_per_epoch
        train_loss_y = train_loss_y/trainer.itr_per_epoch
        train_loss_z = train_loss_z/trainer.itr_per_epoch
        print("... avg train loss for epoch %d: loss=%.4f, loss_x=%.2f, loss_y=%.2f, loss_z=%.2f" \
                % (epoch, train_loss_per_epoch, train_loss_x, train_loss_y, train_loss_z))
        writer.add_scalar('scalar/train', train_loss_per_epoch, epoch)

        ### 4.save model ###
        if epoch % cfg.snapshot_iter == 0 or epoch == cfg.end_epoch - 1:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)


        ### 2.val ###
        '''
        # use trainer but not tester
        val_loss_per_epoch, val_loss_x, val_loss_y, val_loss_z = 0.0, 0.0, 0.0, 0.0
        pbar_test = tqdm(range(trainer.val_itr_per_epoch), file=sys.stdout,bar_format=bar_format)
        for itr in pbar_test:
            try:
                joint_cam_noise_pair, joint_cam_pair, joint_vis_pair, joints_have_depth, lengths= next(trainer.val_iterator[0])
            except StopIteration:
                trainer.val_iterator[0] = iter(trainer.val_batch_generator[0])
                joint_cam_noise_pair, joint_cam_pair, joint_vis_pair, joints_have_depth, lengths= next(trainer.val_iterator[0])
            target = {'coord_cam': joint_cam_pair.cuda(), 'vis': joint_vis_pair.cuda(), 'have_depth': joints_have_depth.cuda()}

            # forward
            # calcul loss
            loss_coord, loss_x, loss_y, loss_z = trainer.model(joint_cam_noise_pair.cuda(), lengths.cuda(), target)
            loss_coord = loss_coord.mean()
            loss_x, loss_y, loss_z = loss_x.mean(), loss_y.mean(), loss_z.mean()

            print_str = '... Val epoch{}/{}'.format(epoch, cfg.end_epoch) \
                    + ' Iter{}/{}:'.format(itr, trainer.val_itr_per_epoch) \
                    + ' loss_coord=%.4f' % (loss_coord.detach())\
                    + ' loss_x=%.2f' % (loss_x.detach())\
                    + ' loss_y=%.2f' % (loss_y.detach())\
                    + ' loss_z=%.2f' % (loss_z.detach())
            pbar_test.set_description(print_str, refresh=False)

            val_loss_per_epoch += loss_coord.detach()
            val_loss_x += loss_x.item()
            val_loss_y += loss_y.item()
            val_loss_z += loss_z.item()

        val_loss_per_epoch = val_loss_per_epoch/trainer.val_itr_per_epoch
        val_loss_x = val_loss_x/trainer.val_itr_per_epoch
        val_loss_y = val_loss_y/trainer.val_itr_per_epoch
        val_loss_z = val_loss_z/trainer.val_itr_per_epoch
        print("... avg val loss for epoch %d: loss=%.4f, loss_x=%.2f, loss_y=%.2f, loss_z=%.2f" \
                % (epoch, val_loss_per_epoch, val_loss_x, val_loss_y, val_loss_z))
        writer.add_scalar('scalar/val', val_loss_per_epoch, epoch)
        #writer.add_scalars('scalar/val_whz',{'loss_w':val_loss_x, 'loss_h':val_loss_y, 'loss_z':val_loss_z} ,epoch)
        '''
        ### 3.test ###
        if args.test:
            tester = Tester(epoch)
            tester._make_batch_generator()

            #tester._make_model()
            model = get_model(pretrained = False).cuda()
            model.load_state_dict(trainer.model.state_dict())
            model.eval()

            preds = []
            #preds_copain = []
            with torch.no_grad():
                for itr, inp in enumerate(tqdm(tester.batch_generator)):
                    joint_cam_noise_pair, lengths = inp
                    coord_out = model(joint_cam_noise_pair.float().cuda(), lengths.cuda())
                    coord_out = coord_out.cpu().numpy()
                    coord_out_0 = coord_out[:,:17]
                    #coord_out_1 = coord_out[:,17:]
                    preds.append(coord_out_0)
                    #preds_copain.append(coord_out_1)
            preds = np.concatenate(preds, axis=0)
            tot_err, tot_err_pa = tester.testset.evaluate_cam(preds)#, preds_copain)
            writer.add_scalars('scalar/test',{'mpjpe':tot_err, 'PA_mpjpe':tot_err_pa} ,epoch)
            writer.close()


if __name__ == "__main__":
    main()
