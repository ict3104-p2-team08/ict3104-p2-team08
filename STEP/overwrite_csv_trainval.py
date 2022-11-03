# To overwrite new_val.csv and new_train.csv with data selected by user

import csv
from config import parse_config

args = parse_config()

selected_train_vid = args.video_train.split(",")
selected_val_vid = args.video_test.split(",")


selected_val_vid_data = []
selected_train_vid_data = []

# get selected val vid data
with open('./datasets/ava/label/ava_val_v2.1.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row[0] in selected_val_vid:  # if row contain selected vid data
            selected_val_vid_data.append(row)

# get selected train vid data
with open('./datasets/ava/label/ava_train_v2.1.csv', 'r') as f2:
    reader = csv.reader(f2, delimiter=',')
    for row in reader:
        if row[0] in selected_train_vid:  # if row contain selected vid data
            selected_train_vid_data.append(row)

# write selected val vid data to new_val.csv
with open('./datasets/ava/label/new_val.csv', 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the data
    for vid_data in selected_val_vid_data:
        writer.writerow(vid_data)

# write selected val vid data to new_val.csv
with open('./datasets/ava/label/new_train.csv', 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the data
    for vid_data in selected_train_vid_data:
        writer.writerow(vid_data)