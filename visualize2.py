# -*- coding: utf-8 -*-
from Dataset.data_generation import SimpleKeyCorridor,collect_positive_data
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import tensor_to_np,make_dir, display_frames_as_gif, detect_peaks, display_ConfidenceScore_as_png, load_h5_data
from PIL import ImageFont, ImageDraw, Image
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from Task_Decomposition.task_decomposition import Decomposition
from Models.model_clip import CLIP_FeatureExtractor
from Models.model_r3m import R3M_FeatureExtractor
from Models.model import Detector


def Visualize(args, png_save_dir, gif_save_dir):
    if args.env == 'maniskill2':
        import h5py
        from mani_skill2.utils.io_utils import load_json
        dataset_file = f"./Dataset/maniskill2/v0/rigid_body/{args.env_id}/trajectory.{args.obs_mode}.{args.control_mode}.h5"
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]

        load_count = len(episodes)
        eps_id = np.random.randint(0, load_count)
        eps = episodes[eps_id]
        trajectory = data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        # TODO Whether use hand_camera
        observations = trajectory["obs"]["image"]["base_camera"]["rgb"][:]
        
        frames_full = observations.copy()
        observations = np.array(observations)
        length = torch.tensor([observations.shape[0]])

        test_data = observations

    # image preprocess
    test_data = np.transpose(test_data,[0,3,1,2])
    test_data = test_data / 255.
    test_data = torch.tensor(test_data).float().to(args.device)
    test_data = pad_sequence([test_data], batch_first=True, padding_value=0)

    # Load model
    if args.visual_representation == "clip":
        embedding_policy = CLIP_FeatureExtractor(model_name='openai/clip-vit-base-patch32', device='cuda')
        input_channel = 512
    elif args.visual_representation == "r3m":
        embedding_policy = R3M_FeatureExtractor(model_name='resnet18', device='cuda')
        input_channel = embedding_policy.hidden_dim
    detector = Detector(input_channel=input_channel)
    state_dict = torch.load(args.model_path)
    detector.load_state_dict(state_dict)
    detector.to(args.device)
    detector.eval()
 
    # Obtain video embdedding and Detect key frame
    emb = embedding_policy(test_data)
    output = detector(emb,length)
    emb = tensor_to_np(emb).squeeze() # (1, len, feature_dim) --> (len, feature_dim)
    output = tensor_to_np(output)[0]

    # key frame candidate = peak + goal
    peak = detect_peaks(output)
    if len(output)-1 not in peak:
        peak = np.append(peak, [len(output)-1])

    # goal decomposition
    task = "Pick up a red cube and place it onto a green one."
    sub_goal, sub_goal_list = Decomposition(args, task)
    k = len(sub_goal_list)
    # numbers = re.findall(r'\d+', sub_goal)
    # k = int(numbers[-1])

    # clustering-v0
    peak_emb = emb[peak]
    y_pred = []
    pred_metrics = []
    for index, gamma in enumerate((0.01, 0.1, 1)):
        y_pred.append(SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(peak_emb))
        pred_metrics.append(metrics.calinski_harabasz_score(peak_emb, y_pred[index]))
    index = np.argmax(pred_metrics)
    y_pred = y_pred[index]

    # semantic alignment
    sim_pred = []
    for t in range(len(y_pred)):
        img_emb = torch.Tensor(peak_emb[t]).to(args.device)
        sim_list = []
        for goal_idx in range(len(sub_goal_list)):
            text_emb = embedding_policy.get_text_feature(sub_goal_list[goal_idx])
            sim_list.append(embedding_policy.txt_to_img_similarity(img_emb, text_emb))
        sim_pred.append(np.argmax(sim_list))
    
    align_idx = sim_pred != y_pred
    for idx in align_idx:
        pass
    
    # save confidence score png
    color = {'red':(255,0,0), 'blue':(0,0,255), 'green':(0,255,0), 'yellow':(255,255,0), 'cyan':(0,255,255)}
    x_axis_data = np.arange(1, len(output)+1)
    y_axis_data = output
    display_ConfidenceScore_as_png(x_axis_data, y_axis_data, png_save_dir, peak, y_pred, color)

    # save gif
    color_values = [value for key,value in color.items()]
    frame_visual = []
    for ind, conf in enumerate(output):
        frame = Image.fromarray(np.uint8(frames_full[ind]))
        draw = ImageDraw.Draw(frame)
        if ind not in peak:
            draw.text((10, 10),  "{:.4f}".format(conf),font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf',size = 25), fill = (255, 255, 255))
        else:
            draw.text((10, 10),  "{:.4f}".format(conf),font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf',size = 25), fill = color_values[y_pred[np.where(peak == ind)][0]])
        frame = np.array(frame)
        frame_visual.append(frame)
    display_frames_as_gif(frame_visual,dir=gif_save_dir)