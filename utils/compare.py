import os
import numpy as np
from utils.utils import get_norm
import pandas as pd

root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"

inputs_list = os.listdir(root_path + "Input/")
inputs_list.sort(key=lambda x: int(x[:-4]))
sum = np.zeros((1, 926))
avg = np.zeros((1, 926))
std = np.zeros((1, 926))

label_data = pd.DataFrame()
label_data = pd.read_csv(root_path + "Label/" + '1.txt', sep=' ', header=None, dtype=float)
label_data = np.array(label_data)

sample = label_data[0]

output_mean, output_std = get_norm("/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/OutputNorm.txt")
scaled = (sample-output_mean) / output_std
dis = scaled - sample

n = 0
for i in dis:
    if i <0:
        n+=1

print(n)
# print((scaled-sample).sum())
