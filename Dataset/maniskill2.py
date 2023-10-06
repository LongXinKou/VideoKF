# -*- coding: utf-8 -*-
import argparse

import gymnasium as gym
import numpy as np
import h5py
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from tqdm.notebook import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

# image trajectory
class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
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
        self.label = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        # for eps_id in tqdm(range(load_count)):
        for eps_id in range(load_count):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # TODO Whether use hand_camera
            self.observations.append(trajectory["obs"]["image"]["base_camera"]["rgb"][:])
            if eps["info"]["success"]:
                self.label.append(1)
            else:
                self.label.append(0)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        frames, label = self.observations[index], self.label[index]
        frames = np.transpose(frames,[0,3,1,2]) # (n,bs,h,w)
        frames = frames / 255.
        return frames, label


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    video_list = []
    label_list = []
    offsets_list = []
    for video, label in data:
        video_list.append(torch.tensor(video))
        offsets_list.append(len(video))
        label_list.append(label)
    video_tensor = pad_sequence(video_list, batch_first=True, padding_value=0)
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    offsets_tensor = torch.tensor(offsets_list)
    return video_tensor, offsets_tensor, label_tensor