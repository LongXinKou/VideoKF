# -*- coding: utf-8 -*-
from Dataset.data_generation import SimpleKeyCorridor,collect_positive_data
from Models.model_cnn import Detector
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import tensor_to_np,make_dir, display_frames_as_gif, detect_peaks, display_ConfidenceScore_as_png, load_h5_data
from PIL import ImageFont, ImageDraw, Image
from torch.nn.utils.rnn import pad_sequence
from Task_Decomposition.task_decomposition import Decomposition


def Visualize(args, png_save_dir, gif_save_dir):
    if args.env == 'Gridworld':
        #generate a unseen environment 
        frames_partial, frames_full = collect_positive_data()

        frames_partial = np.array(frames_partial)
        length = torch.tensor([frames_partial.shape[0]])

        # confidents_matrix = np.zeros((len(frames_partial)))
        # confidents_count = np.zeros((len(frames_partial)))

        test_data = frames_partial
    
    elif args.env == 'miniskill2':
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
    test_data = torch.tensor(test_data).float()
    test_data = pad_sequence([test_data], batch_first=True, padding_value=0)

    detector = Detector()
    state_dict = torch.load(args.model_path)
    detector.load_state_dict(state_dict)
    detector.eval()
    emb = detector.feature_network(test_data)
    emb = emb.detach().numpy().squeeze() # (1, len, feature_dim) --> (len, feature_dim)
    output = detector(test_data,length)
    output = tensor_to_np(output)[0]

    # key frame = key frame + goal
    peak = detect_peaks(output)
    if len(output)-1 not in peak:
        peak = np.append(peak, [len(output)-1])

    task = "Pick up a red cube and place it onto a green one."
    output = Decomposition(args, task)
    # TODO use LLM to generate k
    k = 2

    # clustering-v0
    from sklearn.cluster import SpectralClustering
    from sklearn import metrics
    peak_emb = emb[peak]
    y_pred = []
    pred_metrics = []
    for index, gamma in enumerate((0.01, 0.1, 1)):
        y_pred.append(SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(peak_emb))
        pred_metrics.append(metrics.calinski_harabasz_score(peak_emb, y_pred[index]))
    index = np.argmax(pred_metrics)
    y_pred = y_pred[index]
    
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