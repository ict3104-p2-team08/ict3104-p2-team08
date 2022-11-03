from os import listdir
from os.path import isfile, join
import os
import csv

mypath = "C:\\Users\\leech\\Downloads\\AVA_videos"
trainvidpath = "C:\\Users\\leech\\Downloads\\train_vid"
valvidpath = "C:\\Users\\leech\\Downloads\\val_vid"


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

train_vid, val_vid = [], []
with open('./datasets/ava/label/ava_train_v2.1.csv', 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row[0] not in train_vid:
            train_vid.append(row[0])

with open('./datasets/ava/label/ava_val_v2.1.csv', 'rt') as f2:
    reader = csv.reader(f2, delimiter=',')
    for row in reader:
        if row[0] not in val_vid:
            val_vid.append(row[0])

print(train_vid)
print("")
print(val_vid)

for file in onlyfiles:
    newfilename = file.replace('.mp4', '')
    newfilename = newfilename.replace('.mkv', '')
    newfilename = newfilename.replace('.webm', '')
    if newfilename in train_vid:
        os.replace(mypath + "\\" + file, trainvidpath + "\\" + file)
        print("move " + file + " to train folder")
    elif newfilename in val_vid:
        os.replace(mypath + "\\" + file, valvidpath + "\\" + file)
        print("move " + file + " to val folder")
    else:
        print(file + " does not belong anywhere")
        continue
