import os
import numpy as np
from tqdm import tqdm
import pandas as pd

root_path = "/home/rr/Downloads/nsm_data/Train/"

inputs_list = os.listdir(root_path + "Input/")
inputs_list.sort(key=lambda x: int(x[:-4]))
n = 0
files_num = 2
train_input_data = pd.DataFrame()
train_label_data = pd.DataFrame()
f1 = open(root_path+"Input.txt")

for i, file in enumerate(tqdm(inputs_list)):
    # if i > 10:
    #     break
    if i % files_num or i == 0:

        single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
        single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
        train_input_data = train_input_data.append(single_input_data, ignore_index=True)
        train_label_data = train_label_data.append(single_label_data, ignore_index=True)

    elif (i != 0 and i % files_num == 0) or i == len(inputs_list)-1:

        print(train_label_data.iloc[0, :])
        for xx in range(len(train_label_data)):
            f1.write(train_label_data.iloc[xx, :])

        f1.close()

        train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
        train_label_data = train_label_data.drop(train_label_data.index, inplace=False)

# train_input_data.to_csv(root_path+"Input.txt", header=None, index=None, sep=' ')
# train_label_data.to_csv(root_path+"Output.txt", header=None, index=None, sep=' ')

