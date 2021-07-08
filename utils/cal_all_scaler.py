import os
import numpy as np
from tqdm import tqdm
import pandas as pd

root_path = "/home/rr/Downloads/nsm_data/Train/"

inputs_list = os.listdir(root_path + "Input/")
inputs_list.sort(key=lambda x: int(x[:-4]))
files_num = 50
sum = np.zeros((1, 5307))
n = 0
train_input_data = pd.DataFrame()
train_label_data = pd.DataFrame()

for i, file in enumerate(tqdm(inputs_list)):
    if i % files_num or i == 0:
        single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
        single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
        train_input_data = train_input_data.append(single_input_data, ignore_index=True)
        train_label_data = train_label_data.append(single_label_data, ignore_index=True)

    elif (i != 0 and i % files_num == 0) or i == len(inputs_list)-1:
        input_data = np.array(train_input_data)
        single_sum = input_data.sum(axis=0)

        sum += single_sum
        n += input_data.shape[0]

        train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
        train_label_data = train_label_data.drop(train_label_data.index, inplace=False)

# avg
input_avg = sum/float(n)

std_sum = np.zeros((1, 5307))
for i, file in enumerate(tqdm(inputs_list)):
    if i % files_num or i == 0:
        single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
        single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
        train_input_data = train_input_data.append(single_input_data, ignore_index=True)
        train_label_data = train_label_data.append(single_label_data, ignore_index=True)

    elif (i != 0 and i % files_num == 0) or i == len(inputs_list) - 1:
        input_data = np.array(train_input_data)
        squ = [(item-input_avg)**2 for item in input_data]
        squ = np.array(squ).reshape((-1, 5307))

        std_sum += squ.sum(axis=0)

        train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
        train_label_data = train_label_data.drop(train_label_data.index, inplace=False)

# std
input_std = np.sqrt(std_sum/float(n-1))

iuput_norm = np.array([input_avg[0], input_std[0]])
np.savetxt('CalAllInputNorm.txt', iuput_norm, fmt="%f")

# input_mean, input_std = get_norm("/home/rr/Downloads/nsm_data/Train/InputNorm.txt")
# loss = abs(input_mean - avg)
# print(loss.sum())
