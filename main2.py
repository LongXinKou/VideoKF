# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import time
import argparse
from tqdm import *
from utils import *
from tensorboardX import SummaryWriter

from train import train_epoch,test_epoch
from Dataset.GridWorld import collate_fn
from Dataset.maniskill2 import ManiSkill2Dataset

from Models.model_clip import CLIP_FeatureExtractor
from Models.model_r3m import R3M_FeatureExtractor
from Models.model import Detector, Predictor
from visualize2 import Visualize

parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
# ====Experiment====
parser.add_argument('--train', default="True", type=str,
                    help='Model training or visualizing')
parser.add_argument('--device', default="cuda", type=str)
# ====Environment Setting====
parser.add_argument('--env', default='maniskill2', type=str,
                    help='Environment name')
parser.add_argument('--env_id', default='StackCube-v0', type=str,
                    help='Task of environment')
parser.add_argument('--obs_mode', default='rgbd', type=str,
                    help='Mode of observation, including state and rgbd')
parser.add_argument('--control_mode', default='pd_ee_delta_pose', type=str,
                    help='Mode of robot control')

# ====Key Frame Parameters====
parser.add_argument('--visual_representation', default='clip', type=str,
                    help='clip/r3m')
parser.add_argument('--l1_weight', default=1e-1, type=float,
                    help='Compactness regularization')
parser.add_argument('--classify_weight', default=1, type=float,
                    help='formal classification loss')
parser.add_argument('--reverse_weight', default=2, type=float,
                    help='reverse classification loss')
parser.add_argument('--train_dir', default='./GridWorld/toy_dataset/train', type=str,
                    help='training dir')
parser.add_argument('--test_dir', default='./GridWorld/toy_dataset/test', type=str,
                    help='test dir')
parser.add_argument('--max_Epoch',default=20, type=int, help='The maximum Epochs for learn')
parser.add_argument('--batch_size', default=32, type=int, help='The batch_size for training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--save_model_every_n_steps',default=10, type=int, help='The frequency for saving model')

# ====Visualize Setting====
parser.add_argument('--model_path', default='./', type=str,
                    help='the path for the pretrained model weight')
parser.add_argument('--save_dir', default='./tmp_test', type=str,
                    help='the path to save the visualization result')
parser.add_argument('--model', default='', type=str,
                    help='pretrained model name')

args = parser.parse_args()


if __name__ == '__main__':    
    make_dir(args)

    if args.train == "True":
        if args.env == 'maniskill2':
            # TODO modify dataset(too large)
            train_dataset = ManiSkill2Dataset(f"./Dataset/maniskill2/v0/rigid_body/{args.env_id}/trajectory.{args.obs_mode}.{args.control_mode}.h5")
            args.save_dir = './Weight/' + args.env + '/' + args.env_id
            args.log_dir = './result' + args.env + '/' + args.env_id
        print("Dataset Loaded!")
        train_loader = DataLoader(
                train_dataset,
                batch_size = args.batch_size,
                shuffle = True,
                num_workers = 4,
                pin_memory = True,
                drop_last = False,
                collate_fn=collate_fn,
        )

        # initialize model
        if args.visual_representation == "clip":
            embedding_policy = CLIP_FeatureExtractor(model_name='openai/clip-vit-base-patch32', device='cuda')
            input_channel = 512
        elif args.visual_representation == "r3m":
            embedding_policy = R3M_FeatureExtractor(model_name='resnet18', device='cuda')
            input_channel = embedding_policy.hidden_dim
        critical_state_detector = Detector(input_channel=input_channel)
        return_predictor = Predictor(input_channel=input_channel)

        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            critical_state_detector.cuda()
            return_predictor.cuda()
            criterion.cuda()

        optimizer_detector = torch.optim.Adam(critical_state_detector.parameters(), args.lr,
                                weight_decay=args.weight_decay)

        optimizer_predictor = torch.optim.Adam(return_predictor.parameters(), args.lr,
                                weight_decay=args.weight_decay)

        work_dir = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        

        for epoch in range(args.max_Epoch):
            critical_state_detector.train()
            return_predictor.train()
            acc_detector_mask, acc_predictor = train_epoch(args,
                                            train_loader,
                                            criterion,
                                            optimizer_detector,
                                            optimizer_predictor,
                                            epoch,
                                            critical_state_detector,
                                            return_predictor,
                                            embedding_policy=embedding_policy)


            if epoch % args.save_model_every_n_steps == 0:
                save_checkpoint(
                    state = critical_state_detector.state_dict(), is_best=False, step = epoch, args=args,name='detector_', work_dir=work_dir
                )


                save_checkpoint(
                    state = return_predictor.state_dict(), is_best=False, step = epoch, args=args,name='predictor_', work_dir=work_dir
                )
            print(f"epoch{epoch+1} finished")
    else:
        png_save_dir = args.save_dir + '/' + args.model + '.png'
        gif_save_dir = args.save_dir + '/' + args.model + '.gif'
        Visualize(args, png_save_dir, gif_save_dir)