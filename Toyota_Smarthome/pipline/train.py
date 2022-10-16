from __future__ import division
import time
import os
import argparse
import sys
import torch
import warnings
from tqdm import tqdm
import csv
import wandb

warnings.filterwarnings('ignore')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)', default='rgb')  # added default parameter
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='TSU')  # change default from "charades" to "TSU"
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')  # change default from "50" to "50"
parser.add_argument('-model', type=str, default='PDAN')  # change default from "" to "PDAN"
parser.add_argument('-APtype', type=str, default='map')  # change default from "wap" to "map"
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str, default='False')  # change default from "False" to "True"
parser.add_argument('-num_channel', type=str,
                    default='3')  # change default from "False" to "2" (just random no idea why 2)
parser.add_argument('-batch_size', type=str, default='1')  # change default from "False" to "1"
parser.add_argument('-kernelsize', type=str, default='False')
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')
parser.add_argument('-video_train_test', type=str)
parser.add_argument('-name', type=str)
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# set random seed
if args.randomseed == "False":
    SEED = 0
elif args.randomseed == "True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math

if str(args.APtype) == 'map':
    from apmeter import APMeter

batch_size = int(args.batch_size)

if args.dataset == 'TSU':
    split_setting = str(args.split_setting)

    from smarthome_i3d_per_video import TSU as Dataset
    from smarthome_i3d_per_video import TSU_collate_fn as collate_fn

    classes = 51

    if split_setting == 'CS':
        train_split = './Toyota_Smarthome/pipline/data/smarthome_CS_51.json'
        test_split = './Toyota_Smarthome/pipline/data/smarthome_CS_51.json'

    elif split_setting == 'CV':
        train_split = './Toyota_Smarthome/pipline/data/smarthome_CV_51.json'
        test_split = './Toyota_Smarthome/pipline/data/smarthome_CV_51.json'

    rgb_root = './Toyota_Smarthome/pipline/data/RGB_i3d_16frames_64000_SSD'
    skeleton_root = '/skeleton/feat/Path/'  #

    rgb_root = './Toyota_Smarthome/pipline/data/RGB_v_iashin'


activityList = ["Enter", "Walk", "Make_coffee", "Get_water", "Make_coffee", "Use_Drawer", "Make_coffee.Pour_grains", "Use_telephone",
       "Leave", "Put_something_on_table", "Take_something_off_table",  "Pour.From_kettle",  "Stir_coffee/tea", "Drink.From_cup", "Dump_in_trash",  "Make_tea",
       "Make_tea.Boil_water", "Use_cupboard",  "Make_tea.Insert_tea_bag", "Read", "Take_pills", "Use_fridge", "Clean_dishes",  "Clean_dishes.Put_something_in_sink",
        "Eat_snack", "Sit_down", "Watch_TV", "Use_laptop", "Get_up",  "Drink.From_bottle",  "Pour.From_bottle",  "Drink.From_glass",
        "Lay_down",  "Drink.From_can", "Write", "Breakfast", "Breakfast.Spread_jam_or_butter", "Breakfast.Cut_bread", "Breakfast.Eat_at_table",  "Breakfast.Take_ham",
        "Clean_dishes.Dry_up", "Wipe_table", "Cook",  "Cook.Cut",  "Cook.Use_stove", "Cook.Stir", "Cook.Use_oven", "Clean_dishes.Clean_with_water",
       "Use_tablet",  "Use_glasses", "Pour.From_can"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data_rgb_skeleton(train_split, val_split, root_skeleton, root_rgb):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root_skeleton, root_rgb, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 8
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 2

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


def load_data_rgb(train_split, val_split, root):
    # Load rgb Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


def load_data(train_split, val_split, root):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    # Initialize WandB
    wandb.init(name='Data visualisation',
               project='ICT3104_project',
               notes='This is a testing project',
               tags=['TSU dataset', 'Test Run'])

    bestModel = None
    best_map = 0.0
    loop = tqdm(total=num_epochs, leave=False)
    for model, gpu, dataloader, optimizer, sched, model_file in models:
        num_train_videos = len(dataloader['train'])
        num_test_videos = len(dataloader['val'])
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map, avg_class_prediction = val_step(model, gpu, dataloader['val'], epoch)
            probs.append(prob_val)
            sched.step(val_loss)

            if best_map < val_map:
                best_map = val_map
                bestModel = model
                # prepare and write model performance to csv
                prepare_write_data_for_csv(prob_val, avg_class_prediction, train_map, train_loss, num_train_videos, val_map, val_loss, num_test_videos, epoch)
                # torch.save(model.state_dict(),
                # './Toyota_Smarthome/pipline/' + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))
                # torch.save(model, './Toyota_Smarthome/pipline/' + str(args.model) + '/model_epoch_' + str(args.lr) + '_' + str(epoch))
                # print('save here:', './Toyota_Smarthome/pipline/' + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))

        avg_class_prediction = avg_class_prediction.tolist()
        avg_class_prediction_result = {}
        # actual activity data
        for activity_index in range(len(avg_class_prediction)):
            pred_in_percentage = float_to_percent(avg_class_prediction[activity_index])
            avg_class_prediction_result[activityList[activity_index]] = pred_in_percentage
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Average Class Prediction": avg_class_prediction_result,
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc": train_map,
            "Valid Loss": val_loss,
            "Valid Acc": val_map,})

        # show progress bar
        loop.set_description("training..")
        loop.update(1)
    torch.save(bestModel, './Toyota_Smarthome/pipline/models/' + str(args.name))
    print("Completed")


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)

    outputs_final = activation

    if args.model == "PDAN":
        # print('outputs_final1', outputs_final.size())
        outputs_final = outputs_final[:, 0, :, :]
    # print('outputs_final',outputs_final.size())
    outputs_final = outputs_final.permute(0, 2, 1)
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    loss = loss_f

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()
    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    # print('train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    # print('val-map:', val_map)
    # print("apm value: ")
    # print(100 * apm.value())
    avg_class_prediction = apm.value()
    apm.reset()

    return full_probs, epoch_loss, val_map, avg_class_prediction


# loop through each tested videos to generate output data for saving into csv
def prepare_write_data_for_csv(prob_val, avg_class_prediction, train_map, train_loss, num_train_videos, val_map, val_loss, num_test_videos, epoch):
    dictForMaxAndIndex = {}
    for video, video_value in prob_val.items():
        arrayForMaxAndIndex = []
        for video_length in range(len(prob_val.get(video)[0])):
            activityAtEachFrameArray = []
            for activity_index in range(len(prob_val.get(video))):
                activityAtEachFrameArray.append(prob_val.get(video)[activity_index][video_length])
            highest_confident = max(activityAtEachFrameArray)
            highest_activity_index = activityAtEachFrameArray.index(highest_confident)
            highest_confident = str(float_to_percent(highest_confident)) + "%"
            arrayForMaxAndIndex.append([activityList[highest_activity_index], highest_confident])

        # split arrayForMaxAndIndex into start and end frames with same activity and confident that occur consecutively
        activity_frames_video_accuracy_array = []
        start_array_position = 0
        for vid_length in range(len(arrayForMaxAndIndex)):
            if vid_length != len(arrayForMaxAndIndex) - 1: # if not at last item
                current_activity = arrayForMaxAndIndex[vid_length][0]
                current_confident_level = arrayForMaxAndIndex[vid_length][1]
                next_length_activity = arrayForMaxAndIndex[vid_length + 1][0]
                next_length_confident_level = arrayForMaxAndIndex[vid_length + 1][1]
                if current_activity == next_length_activity and current_confident_level == next_length_confident_level:
                    continue
                else:
                    activity_frames_video_accuracy_array.append([current_activity, (start_array_position * 16) + 1, (vid_length + 1) * 16, video, current_confident_level])
                    start_array_position = vid_length + 1
            else: # last item
                if start_array_position != vid_length: # current item same as prev
                    activity_frames_video_accuracy_array.append([arrayForMaxAndIndex[vid_length][0], (start_array_position * 16) + 1, (vid_length + 1) * 16, video, arrayForMaxAndIndex[vid_length][1]])
                else:
                    activity_frames_video_accuracy_array.append([arrayForMaxAndIndex[vid_length][0], (vid_length * 16) + 1, (video_length + 1) * 16, video, arrayForMaxAndIndex[vid_length][1]])

        dictForMaxAndIndex[video] = activity_frames_video_accuracy_array
    # all torch tensor class type
    print("type of apm", avg_class_prediction)
    avg_class_prediction = avg_class_prediction.tolist()
    print("type of train map", train_map)
    train_map = float(train_map)
    print("type of train loss", train_loss)
    train_loss = float(train_loss)
    print("type of val map", val_map)
    val_map = float(val_map)
    print("type of val loss", val_loss)
    val_loss = float(val_loss)

    # write to csv in output folder
    with open("./Toyota_Smarthome/pipline/result/" + args.name + ".csv", "w", newline="") as file:

        # create write object
        writer = csv.writer(file)

        # create header 1
        header_1 = ["Activity", "Average Class Prediction"]
        writer.writerow(header_1)

        # actual activity data
        for activity_indx in range(len(avg_class_prediction)):
            pred_in_percentage = float_to_percent(avg_class_prediction[activity_indx])
            writer.writerow([activityList[activity_indx], str(pred_in_percentage) + "%"])

        # create header 2
        header_2 = ["Trained on", "Train m-AP", "Train loss", "Tested on", "Prediction m-AP", "Prediction loss", "Epoch"]
        writer.writerow(header_2)

        # write content 2
        train_map = convert_two_decimal(train_map)
        train_loss = convert_two_decimal(train_loss)
        val_map = convert_two_decimal(val_map)
        val_loss = convert_two_decimal(val_loss)
        writer.writerow([str(num_train_videos) + " TSU videos", str(train_map) + "%", str(train_loss), str(num_test_videos) + " TSU videos", str(val_map) + "%", str(val_loss), epoch])

        # write header 3
        header_3 = ["Event", "Start_frame", "End_frame", "Video_Name", "Prediction Accurary for the video"]
        writer.writerow(header_3)

        # write content 3
        for video, result_data in dictForMaxAndIndex.items():
            for row_data in result_data:
                writer.writerow(row_data)

    #print(dictForMaxAndIndex)


def convert_two_decimal(num):
    return num - num % 0.0001


# function to convert float num to percentage value
def float_to_percent(num):
    num = convert_two_decimal(num) * 100
    if num % 1 == 0:
        return int(num)
    else:
        return num


def filter_json_file(list_to_filter):
    f = open(test_split)
    data = json.load(f)
    dict_you_want = {your_key + "_rgb": data[your_key] for your_key in list_to_filter}
    with open("./Toyota_Smarthome/pipline/data/" + args.name + "_CS.json", "w") as f:
        json.dump(dict_you_want, f)


if __name__ == '__main__':
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())

    video_list = args.video_train_test.split(",")

    filter_json_file(video_list)

    train_split = './Toyota_Smarthome/pipline/data/' + args.name + '_CS.json'
    test_split = './Toyota_Smarthome/pipline/data/' + args.name + '_CS.json'

    if args.mode == 'flow':
        pass  # ownself added this line to prevent error
        # print('flow mode', flow_root) #ownself commented
        # dataloaders, datasets = load_data(train_split, test_split, flow_root) #ownself commented
    elif args.mode == 'skeleton':
        print('Pose mode', skeleton_root)
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        #dataloaders, datasets = load_data(train_split, test_split, rgb_root)
        dataloaders, datasets = load_data_rgb(train_split, test_split, rgb_root)

    if args.train:
        num_channel = args.num_channel
        if args.mode == 'skeleton':
            input_channnel = 256
        else:
            input_channnel = 1024

        num_classes = classes
        mid_channel = int(args.num_channel)

        if args.model == "SSPDAN":
            print("you are processing SSPDAN")
            from models import SSPDAN as Net

            model = Net(num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)
        else:
            print("you are processing PDAN")
            from models import PDAN as Net

            model = Net(num_stages=1, num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)

        model = torch.nn.DataParallel(model)

        if args.load_model != "False":
            # entire model
            model = torch.load(args.load_model)
            # weight
            # model.load_state_dict(torch.load(str(args.load_model)))
            print("loaded", args.load_model)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('num_channel:', num_channel, 'input_channnel:', input_channnel, 'num_classes:', num_classes)
        model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        print(lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))
