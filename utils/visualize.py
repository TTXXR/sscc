import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


root_path = "D:/nsm_data/bone_gating_WalkTrain/"
inputs_list = os.listdir(root_path + "Input/")
inputs_list.sort(key=lambda x: int(x[:-4]))
files_num = 10
n = 0

train_input_data = pd.DataFrame()
train_label_data = pd.DataFrame()

for i, file in enumerate(tqdm(inputs_list)):
    # if i % files_num or i == 0:
    if i < len(inputs_list) - 1:
        single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
        single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
        train_input_data = train_input_data.append(single_input_data, ignore_index=True)
        train_label_data = train_label_data.append(single_label_data, ignore_index=True)

    # elif (i != 0 and i % files_num == 0) or i == len(inputs_list)-1:
    elif i == (len(inputs_list) - 1):
        input_data = np.array(train_input_data)
        index = [i for i in range(len(input_data))]

        err_list = []
        zero_list = []
        for i in range(train_input_data.shape[1]):
            err = input_data[:, i].max() - input_data[:, i].min()
            err_list.append(err)
            if err == 0:
                zero_list.append(err)

        print("all 0:", zero_list, "  len：", err_list.count(0))

        col = 3
        print(col, "-max:", input_data[:, col].max(), "  ", col, "-min:", input_data[:, col].min())
        plt.plot(index, input_data[:, col], 'y*-')
        plt.axis([0, 100, input_data[:, col].min(), input_data[:, col].max()])
        plt.show()

        # train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
        # train_label_data = train_label_data.drop(train_label_data.index, inplace=False)
