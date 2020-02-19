import os
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

eval_list = []

def process_file(in_file, label_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    label_dict = {}
    with open(label_file, 'r') as f:
        label_inf = f.readlines()
        for val in label_inf:
            label_dict[val.split()[0]] = float(val.split()[-1])

    live_dict = dict()
    spoof_dict = dict()
    for line in lines:
        data = line.strip().split(' ')
        if len(data) != 2:
            continue
        img_path = data[0]
        # label = float(data[-3])
        liveness = float(data[-1])
        track_id = '/'.join(img_path.split('/')[:-2])
        try:
            label = label_dict[track_id]
        except:
            print(track_id)
            print(label_dict['dev/003000'])

        if label == 1.0:
            if track_id not in live_dict:
                live_dict[track_id] = []
            live_dict[track_id].append(liveness)
        elif label == 0.0:
            if track_id not in spoof_dict:
                spoof_dict[track_id] = []
            spoof_dict[track_id].append(liveness)
        else:
            raise ValueError

    return live_dict, spoof_dict