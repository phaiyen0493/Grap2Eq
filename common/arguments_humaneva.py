# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='humaneva15', type=str,
                        metavar='NAME', help='target dataset')  # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='detectron_pt_coco',
                        type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='Train/S1,Train/S2,Train/S3', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='Validate/S1,Validate/S2,Validate/S3',
                        type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-in-chans', '--in-chans', default=2, type=int,
                        metavar='N', help='input dimension')
    parser.add_argument('--save_lmin', default=10, type=float, metavar='N')
    parser.add_argument('--save_lmax', default=13.5, type=float, metavar='N')
    parser.add_argument('--save_emin', default=1, type=int, metavar='N')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint/model_humaneva', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-cf', '--checkpoint-frequency', default=1, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--nolog', action='store_true',
                        help='forbiden log function')
    parser.add_argument('--evaluate', default='', type=str,
                        metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true',
                        help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true',
                        help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true',
                        help='save training curves as .png images')

    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int,
                        metavar='N', help='dataset interval strides to use during training')
    parser.add_argument('-e', '--epochs', default=60, type=int,
                        metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='batch size')
    parser.add_argument('-drop', '--dropout', default=0.,
                        type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.00006,
                        type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.993, type=float,
                        metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('--coverlr', action='store_true',
                        help='cover learning rate with assigned during resuming previous model')
    parser.add_argument('-mloss', '--min_loss', default=100000, type=float,
                        help='assign min loss(best loss) during resuming previous model')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-cs', default=512, type=int,
                        help='channel size of model for trasformer, needs to be divided by 8')
    parser.add_argument('-tem-alpha', '--temporal-alpha', default=0.25, type=float,
                        metavar='N', help='temporal alpha to decide the rate of decreasing for temporal graph attn')
    parser.add_argument('-tem-beta', '--temporal-beta', default=1, type=int,
                        metavar='N', help='temporal beta to decide the rate of decreasing for temporal graph attn')
    parser.add_argument('-dep', default=8, type=int, help='depth of model')
    parser.add_argument('-alpha', default=0.01, type=float,
                        help='used for wf_mpjpe')
    parser.add_argument('-beta', default=2, type=float,
                        help='used for wf_mpjpe')
    parser.add_argument('--postrf', action='store_true',
                        help='use the post refine module')
    parser.add_argument('--ftpostrf', action='store_true',
                        help='For fintune to post refine module')
    parser.add_argument('-f', '--number-of-frames', default= 27, type=int, metavar='N',
                        help='how many frames used as input')
    parser.add_argument('-lite', '--lite-ver', action='store_true', default=False,
                        help='get light version without CLIP fine grained embeddings')
    parser.add_argument('-h36mpre', '--h36m-pretrain', action='store_true', default=False,
                        help='get light version without CLIP fine grained embeddings')

    # Experimental
    parser.add_argument('-gpu', default='0', type=str,
                        help='assign the gpu(s) to use')
    parser.add_argument('--subset', default=1, type=float,
                        metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int,
                        metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true',
                        help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true',
                        help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true',
                        help='disable optimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true',
                        help='use only linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true',
                        help='disable projection for semi-supervised setting')
    parser.add_argument('--ft', action='store_true',
                        help='use ft 2d(only for detection keypoints!)')
    parser.add_argument('--ftpath', default='checkpoint/exp13_ft2d',
                        type=str, help='assign path of ft2d model chk path')
    parser.add_argument('--ftchk', default='epoch_330.pth',
                        type=str, help='assign ft2d model checkpoint file name')
    parser.add_argument('--no_eval', action='store_true',
                        default=False, help='no_eval')

    # Training
    parser.add_argument('--cond_pose_mask_prob', default=0.1,
                        type=float, help='prob to mask the whole pose')
    parser.add_argument('--cond_part_mask_prob', default=0.1,
                        type=float, help='prob to mask the part')
    parser.add_argument('--cond_joint_mask_prob', default=0.1,
                        type=float, help='prob to mask the joint')

    # Visualization
    parser.add_argument('--viz-subject', type=str,
                        metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str,
                        metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0,
                        metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str,
                        metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0,
                        metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH',
                        help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH',
                        help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000,
                        metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true',
                        help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1,
                        metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1,
                        metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5,
                        metavar='N', help='image size')
    parser.add_argument('--compare', action='store_true', default=False,
                        help='Whether to compare with other methods e.g. Poseformer')
    # ft2d.py
    parser.add_argument('-lcs', '--linear_channel_size', type=int,
                        default=1024, metavar='N', help='channel size of the LinearModel')
    parser.add_argument('-depth', type=int, default=4,
                        metavar='N', help='nums of blocks of the LinearModel')
    parser.add_argument('-ldg', '--lr_decay_gap', type=float, default=10000,
                        metavar='N', help='channel size of the LinearModel')

    parser.add_argument('-scale', default=1.0, type=float,
                        help='the scale of SNR')
    parser.add_argument('-timestep', type=int, default=1000,
                        metavar='N', help='timestep')
    parser.add_argument('-sampling_timesteps', type=int,
                        default=1, metavar='N', help='sampling_timesteps')   # 5
    parser.add_argument('-num_proposals', type=int,
                        default=1, metavar='N')  # 300
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debugging mode')
    parser.add_argument('--p2', action='store_true',
                        default=False, help='using protocol #2, i.e., P-MPJPE')

    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args