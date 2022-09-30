#import sys
#sys.path.append('../')
from utils.utils import build_cfg_path
from models.i3d.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import numpy as np
import torch

# Select the feature type
feature_type = 'i3d'
args = OmegaConf.load(build_cfg_path(feature_type))
args.video_paths = ["../data/input_files/P03T10C01.mp4"]
args.flow_type = 'raft'
args.stack_size = 32

extractor = ExtractI3D(args)

# Extract features
for video_path in args.video_paths:
    print(f'Extracting for {video_path}')
    feature_dict = extractor.extract(video_path)
    [(print(k), print(v.shape), print(v)) for k, v in feature_dict.items()]

    rgb = list(feature_dict.items())[0]
    flow = list(feature_dict.items())[1]

    rgb_dict = {rgb[0]: rgb[1]}
    #flow_dict = {flow[0]: flow[1]}
    #rgb_flow_dict = {rgb[0]: rgb[1], flow[0]: flow[1]}

    #rgb .npy
    np.save(video_path.replace('../data/input_files/', '')[:-4]  + "_rgb.npy", rgb[1])

    #flow .npy
    #np.save('P02T03C03' + "_flow.npy", flow_dict)

    #rgb_flow .npy
    #np.save('P02T03C03' + "_rgb_flow.npy", rgb_flow_dict)

    rbg_test =np.load(video_path.replace('../data/input_files/', '')[:-4] + "_rgb.npy", allow_pickle= True)
    #flow_test =np.load("P02T03C03_flow.npy", allow_pickle= True)
    #rbg_flow_test =np.load("P02T03C03_rgb_flow.npy", allow_pickle= True)


    print("rgb_test: ", rbg_test)
    #print(flow_test)
    #print(rbg_flow_test)