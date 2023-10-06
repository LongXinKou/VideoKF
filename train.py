import torch
from torch.autograd import Variable
import sys
import os
from tqdm import *
import numpy as np
import random
from utils import save_checkpoint, tensor_to_np, accuracy


def train_epoch(args,
                train_loader,
                criterion,
                optimizer_detector,
                optimizer_predictor,
                epoch,
                critical_state_detector,
                return_predictor,
                embedding_policy = None):
    param_groups = optimizer_detector.param_groups[0]
    curr_lr = param_groups["lr"]
    msg_dict = {}
    # loss_detector, loss_predictor = [], []
    acc_detector_mask, acc_predictor = [], []
    with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
        for frames,length,labels in tqdmDataLoader:
            frames = frames.float()
            if torch.cuda.is_available():
                frames = frames.cuda()
                labels = labels.cuda()

            optimizer_detector.zero_grad()

            if embedding_policy is not None:

                frames = embedding_policy(frames)
            
            mask = critical_state_detector(frames, length)

            loss_l1 =  args.l1_weight * torch.linalg.norm(mask,ord=1)

            bs,f_len = mask.size()
            if embedding_policy is not None:
                mask = mask.view(bs,f_len,1)
            else:
                mask = mask.view(bs,f_len,1,1,1)
            masked_frames = mask * frames

            # output(bs,2) --> binary classification;   acc_mask --> accuracy of predictor(masked_frames)
            output = return_predictor(masked_frames,length)

            loss_classify = args.classify_weight * criterion(output,labels)
            acc_mask = accuracy(output,labels)[0] 

            reverse_mask = torch.ones_like(mask) - mask
            reverse_frames = reverse_mask * frames

            output_r = return_predictor(reverse_frames,length)
            # TODO why confused_label instead of label?
            # confused_label = torch.ones_like(output_r)*0.5
            # loss_classify_r = args.reverse_weight * criterion(output_r,confused_label)
            loss_classify_r = -1 * args.reverse_weight * criterion(output_r,labels)
            acc_r = accuracy(output_r,labels)[0]


            loss_total = loss_l1 + loss_classify + loss_classify_r


            loss_total.backward()
            optimizer_detector.step()

            optimizer_predictor.zero_grad()

            output = return_predictor(frames,length)

            loss_classify_traj = criterion(output,labels)
            acc_clean = accuracy(output,labels)[0]

            # loss_detector.append(loss_total.data.cpu())
            # loss_predictor.append(loss_classify_traj.data.cpu())
            acc_detector_mask.append(tensor_to_np(acc_mask)[0])
            acc_predictor.append(tensor_to_np(acc_clean)[0])

            loss_classify_traj.backward()
            optimizer_predictor.step()
        return acc_detector_mask, acc_predictor

def test_epoch(args,test_loader,critical_state_detector,return_predictor):
    critical_state_detector.eval()
    return_predictor.eval()
    acc_mask_avg = 0
    acc_r_avg = 0
    acc_clean_avg = 0
    l1_regular_list = []
    var_list = []
    for steps, data in enumerate(test_loader):
        (frames,length,labels) = data
        frames = frames.float()
        if torch.cuda.is_available():
            frames = frames.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            mask = critical_state_detector(frames,length)

            l1_regular_list.append(tensor_to_np(torch.linalg.norm(mask,ord=1)))
            var_list.append(tensor_to_np(torch.var(mask)))
            bs,f_len = mask.size()
            mask = mask.view(bs,f_len,1,1,1)
            masked_frames = mask * frames

            output = return_predictor(masked_frames,length)


            acc_mask = accuracy(output,labels)[0]

            reverse_mask = torch.ones_like(mask) - mask
            
            reverse_frames = reverse_mask * frames
            
            output_r = return_predictor(reverse_frames,length)

            acc_r = accuracy(output_r,labels)[0]

            output = return_predictor(frames, length)

            acc_clean = accuracy(output,labels)[0]

        acc_mask_avg += tensor_to_np(acc_mask)[0]
        acc_r_avg += tensor_to_np(acc_r)[0]
        acc_clean_avg += tensor_to_np(acc_clean)[0]
    var_list = np.array(var_list)
    l1_regular_list = np.array(l1_regular_list)
    
    return acc_mask_avg/(steps+1.0), acc_r_avg/(steps+1.0), acc_clean_avg/(steps+1.0)
