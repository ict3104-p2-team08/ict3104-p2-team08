#import sys
#sys.path.append('../')
from utils.utils import build_cfg_path
from models2.i3d.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import numpy as np
import argparse
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-videosToExtract', type=str)
args = parser.parse_args()

rgb_viashin_path = '../Toyota_Smarthome/pipline/data/i3d_CS.json'
smarthome_cs_path = '../Toyota_Smarthome/pipline/data/smarthome_CS_51.json'

if __name__ == '__main__':
    print("checking for select video's npy")
    video_list = args.videosToExtract.split(",")
    new_video_list = [video + "_rgb" for video in video_list]
    video_to_extract = []
    print(new_video_list)
    f = open(rgb_viashin_path)
    rgb_viashin_json = json.load(f)
    # get videos with feature not extracted before
    for each_video in new_video_list:
        if each_video not in rgb_viashin_json.keys():
            video_to_extract.append(each_video)
    # get values from smarthome_cs_json base on videos not extracted
    video_to_extract_without_rgb_substring = []
    for video in video_to_extract:
        video_to_extract_without_rgb_substring.append(video.replace("_rgb", ""))
    f = open(smarthome_cs_path)
    smarthome_cs_json = json.load(f)
    dict_you_want = {your_key + "_rgb": smarthome_cs_json[your_key] for your_key in video_to_extract_without_rgb_substring}

    # Select the feature type
    feature_type = 'i3d'
    args = OmegaConf.load(build_cfg_path(feature_type))
    args.flow_type = 'raft'
    args.streams = 'rgb'
    args.stack_size = 64
    args.step_size = 64

    extractor = ExtractI3D(args)

    # Extract features
    for video in video_to_extract_without_rgb_substring:
        print(video + "_rgb.mp4 npy file does not exist")
        video = video + ".mp4"

        # check if actual mp4 video exist in data/input_files
        try:
            f = open("../data/input_files/" + video)
            # Do something with the file
        except IOError:
            print(video + " does not exist in data/input_files folder")
            continue

        print(f'Extracting for {video}')
        feature_dict = extractor.extract("../data/input_files/" + video)
        [(print(k), print(v.shape), print(v)) for k, v in feature_dict.items()]

        # derive rgb, flow, rgb+flow dict
        rgb = list(feature_dict.items())[0]
        #flow = list(feature_dict.items())[1]

        rgb_dict = {rgb[0]: rgb[1]}
        #flow_dict = {flow[0]: flow[1]}
        #rgb_flow_dict = {rgb[0]: rgb[1], flow[0]: flow[1]}

        # rgb .npy
        data = np.expand_dims(rgb[1], axis=(2, 1)) # unsqueeze data to fit TSU dimension
        np.save("../Toyota_Smarthome/pipline/data/RGB_v_iashin/" + video[:-4] + "_rgb.npy", data)

        # flow .npy
        #np.save(w.value[:-4] + "_flow.npy", flow_dict)

        # rgb_flow .npy
        #np.save(w.value[:-4] + "_rgb_flow.npy", rgb_flow_dict)

        # check if .npy load
        #rbg_test = np.load(video + "_rgb.npy", allow_pickle=True)
        #flow_test = np.load("P02T03C03_flow.npy", allow_pickle=True)
        #rbg_flow_test = np.load("P02T03C03_rgb_flow.npy", allow_pickle=True)

        #print(rbg_test)
        #print(flow_test)
        #print(rbg_flow_test)

        # update i3d_CS_.json
        with open(rgb_viashin_path) as outfile:
            data = json.load(outfile)
            added_npy = {video[:-4] + "_rgb": dict_you_want[video[:-4] + "_rgb"]}
        data.update(added_npy)

        with open(rgb_viashin_path, 'w') as outfile:
            json.dump(data, outfile)

    #with open(rgb_viashin_path) as outfile:
    #    data = json.load(outfile)
    #data.update(dict_you_want)

    #with open(rgb_viashin_path, 'w') as outfile:
    #    json.dump(data, outfile)

    print("done extraction")