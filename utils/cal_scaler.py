import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils.utils import get_norm

root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"

inputs_list = os.listdir(root_path + "Input/")
inputs_list.sort(key=lambda x: int(x[:-4]))
files_num = 10
sum = np.zeros((1, 926))
avg = np.zeros((1, 926))
std = np.zeros((1, 926))
n = 0

train_input_data = pd.DataFrame()
train_label_data = pd.DataFrame()

for i, file in enumerate(tqdm(inputs_list)):

    if i < len(inputs_list)-1:
    # if i % files_num or i == 0:
        single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
        single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
        train_input_data = train_input_data.append(single_input_data, ignore_index=True)
        train_label_data = train_label_data.append(single_label_data, ignore_index=True)

    # elif (i != 0 and i % files_num == 0) or i == len(inputs_list)-1:
    elif i == (len(inputs_list)-1):
        # input
        t_input_data = np.array(train_input_data)
        # t_label_data = np.array(train_label_data)
        # single_sum = t_input_data.sum(axis=0)
        input_avg = np.mean(t_input_data, axis=0)
        input_std = np.std(t_input_data, axis=0)

        iuput_norm = np.array([input_avg, input_std])
        np.savetxt('InputNorm.txt', iuput_norm, fmt="%f")

        # input
        t_label_data = np.array(train_label_data)
        # t_label_data = np.array(train_label_data)
        # single_sum = t_input_data.sum(axis=0)
        output_avg = np.mean(t_label_data, axis=0)
        output_std = np.std(t_label_data, axis=0)

        output_norm = np.array([output_avg, output_std])
        np.savetxt('OutputNorm.txt', output_norm, fmt="%f")

        # train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
        # train_label_data = train_label_data.drop(train_label_data.index, inplace=False)


input_mean, input_std = get_norm("/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/InputNorm.txt")
input_mean, input_std = input_mean[0:926], input_std[0:926]
loss = abs(input_mean - avg)
print(loss.sum())
