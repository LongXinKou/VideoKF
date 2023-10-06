'''
Robotic Learning Tutorial Part 2: Imitation Learning
'''
import argparse
import os.path as osp

import gymnasium as gym
import numpy as np
import h5py
import torch as th
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from tqdm.notebook import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from torch.utils.data import Dataset, DataLoader

'''
1. download the demonstrations dataset
2. convert the trajectories to add observations back in
3. Setting up the Dataset
'''
# 1. download the demonstrations dataset
# python -m mani_skill2.utils.download_demo rigid_body -o ./Dataset/maniskill2
# python -m mani_skill2.utils.download_demo soft_body -o ./Dataset/maniskill2

# 2. convert the trajectories to add observations back in
'''
python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    Dataset/maniskill2/v0/rigid_body/LiftCube-v0/trajectory.h5 --save-traj \
    --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

import urllib.request
!mkdir -p "demos/v0/rigid_body/LiftCube-v0"
urllib.request.urlretrieve("https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/processed_demos/LiftCube-v0.tar.gz", "demos/v0/rigid_body/LiftCube-v0.tar.gz")
!tar -xvzf "demos/v0/rigid_body/LiftCube-v0.tar.gz" -C "demos/v0/rigid_body/"
'''

# 3. Setting up the Dataset
# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1, obs_mode="rgbd", control_mode="pd_ee_delta_pose") -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        import h5py
        from mani_skill2.utils.io_utils import load_json
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.label = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            if obs_mode == "state":
                self.observations.append(trajectory["obs"][:-1])
            elif obs_mode == "rgbd":
                self.observations.append(trajectory["obs"]["image"]["base_camera"]["rgb"][:-1])
            if eps["info"]["success"]:
                self.label.append(1)
            else:
                self.label.append(0)
            self.actions.append(trajectory["actions"])
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = th.from_numpy(self.actions[idx]).float()
        obs = th.from_numpy(self.observations[idx]).float()
        return obs, action

parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
# ====Experiment====
parser.add_argument('--train', default="True", type=str,
                    help='Model training or visualizing')
# ====Environment Setting====
parser.add_argument('--env', default='gridworld', type=str,
                    help='Environment name')
parser.add_argument('--env_id', default='LiftCube-v0', type=str,
                    help='Task of environment')
parser.add_argument('--obs_mode', default='rgbd', type=str,
                    help='Mode of observation, including state and rgbd')
parser.add_argument('--control_mode', default='pd_ee_delta_pose', type=str,
                    help='Mode of robot control')

# ====Key Frame Parameters====
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
parser.add_argument('--batch_size', default=64, type=int, help='The batch_size for training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--save_model_every_n_steps',default=1, type=int, help='The frequency for saving model')

# ====Visualize Setting====
parser.add_argument('--model_path', default='./', type=str,
                    help='the path for the pretrained model weight')
parser.add_argument('--save_dir', default='./tmp_test', type=str,
                    help='the path to save the visualization result')
parser.add_argument('--model', default='', type=str,
                    help='pretrained model name')

# ====task decomposition setting==== 
parser.add_argument('--llm_model_path', default='./Models/flan-t5-base', type=str,
                    help='pretrained LLM model path')
args = parser.parse_args()

# ===env setting===
# env_id = "LiftCube-v0"
# obs_mode = "rgbd"
# control_mode = "pd_ee_delta_pose"

# dataset = ManiSkill2Dataset(f"./Dataset/maniskill2/v0/rigid_body/{env_id}/trajectory.{obs_mode}.{control_mode}.h5")
# dataloader = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
# obs, action = dataset[0]
# print("Observation:", obs.shape)
# print("Action:", action.shape)

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration
)
import torch
# tokenizer = AutoTokenizer.from_pretrained("./Models/flan-t5-base")

# sequence = "Using a Transformer network is simple"
# tokens = tokenizer.tokenize(sequence)
# tokens_2 = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")
# tokens_3 = torch.tensor(tokenizer.encode(sequence,add_special_tokens=True)).unsqueeze(0) 
# print(tokens) 

# #  从token 到输入 ID
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids) 
f = open("Dataset/prompt_example.txt")
lm_template = f.read()
task = "Pick up a red cube and place it onto a green one."
prompt = "\n".join([lm_template, f"TASK: {task}", "SUBGOALS: "])
print(prompt)

# load model from the hub
tokenizer = T5Tokenizer.from_pretrained(args.llm_model_path)
model = T5ForConditionalGeneration.from_pretrained(args.llm_model_path)
# model = model.to_bettertransformer().cuda()

# inputs = tokenizer(prompt, return_tensors="pt")
# output_sequence = model.generate(input_ids=inputs['input_ids'], max_length=50)
# outputs = tokenizer.batch_decode(output_sequence, skip_special_tokens=True)
# for output in outputs:
#     print(output)

inputs = tokenizer(prompt, return_tensors="pt").input_ids
output_sequence = model.generate(input_ids=inputs, max_length=50)
output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
print(output)
