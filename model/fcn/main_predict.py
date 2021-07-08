import os
import pandas as pd
import numpy as np

import torch
from net import Model
from utils.utils import get_norm
from torch.autograd import Variable


def predict(net, input, label, seq_flag=False):
    if not seq_flag:
        return net.model(input), label
    else:
        out = []
        x = input[0]
        for i in range(len(input)):
            x = net.model(x)
            out.append(x)
            print(x)

        return out, label


if __name__ == '__main__':
    root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"
    # root_path = "/home/rr/Downloads/nsm_data/Train/"

    loss_func = torch.nn.MSELoss()
    num_inputs, num_outputs, num_hiddens = 926, 618, 926*4
    lr, batch_size, num_epochs = 0.1, 64, 40
    # num_inputs, num_outputs, num_hiddens = 5307, 618, 1024
    # lr, batch_size, num_epochs = 0.01, 32, 40

    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    net = Model(num_inputs, num_hiddens, num_outputs, lr, batch_size)
    net.model.load_state_dict(
        torch.load(os.path.join("model_list/part_BN_outscale_3l_4h_models", "fcn_80.pth"), map_location=torch.device("cuda:0")))
    net.optimizer.load_state_dict(
        torch.load(os.path.join("model_list/part_BN_outscale_3l_4h_models", "fcn_opt_80.pth"), map_location=torch.device("cuda:0")))
    net.model.eval()

    input_data = pd.read_csv(root_path + "Input/" + "10.txt", sep=' ', header=None, dtype=float)
    label_data = pd.read_csv(root_path + "Label/" + "10.txt", sep=' ', header=None, dtype=float)

    # nsm_data = pd.read_excel(root_path+"../nsm_data.xlsx", sheet_name="Input")
    print(len(label_data.iloc[0, :]))
    nsm_data = pd.read_csv(root_path + "../nsm_walk_data/Input.txt", sep=' ', header=None, dtype=float)
    nsm_data = nsm_data.iloc[:, 0:5307]
    nsm_data = torch.Tensor(np.array(nsm_data))
    nsm_data = Variable(nsm_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
    # nsm_label_data = pd.read_excel(root_path + "../nsm_data.xlsx", sheet_name="Output")
    nsm_label_data = pd.read_csv(root_path + "../nsm_walk_data/Output.txt", sep=' ', header=None, dtype=float)
    nsm_label_data = nsm_label_data.iloc[:, 0:618]
    nsm_label_data = torch.Tensor(np.array(nsm_label_data))
    nsm_label_data = Variable(nsm_label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

    # scale 标准化
    # scale = StandardScaler()
    # scale = scale.fit(input_data)
    # input_data = torch.Tensor(scale.transform(input_data))

    # 手动 标准化
    input_mean, input_std = get_norm("data/InputNorm.txt")
    output_mean, output_std = get_norm("data/OutputNorm.txt")
    input_mean, input_std = input_mean[0:926], input_std[0:926]
    input_data = torch.Tensor((np.array(input_data).astype('float32') - input_mean) / input_std)

    # input_data = torch.Tensor(np.array(input_data))
    label_data = torch.Tensor(np.array(label_data))
    input_data = Variable(input_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
    label_data = Variable(label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

    zero_list = np.zeros((1, 618))
    zero_list = torch.Tensor(np.array(zero_list))
    zero_list = Variable(zero_list.type(torch.FloatTensor).to(torch.device("cuda:0")))

    loss_sum = 0
    for item in zip(nsm_data, nsm_label_data):
        pred, target = predict(net, item[0][0:926], item[1])
        pred = Variable(torch.Tensor(pred.cpu().detach().numpy() * output_std + output_mean).type(torch.FloatTensor).to(
            torch.device("cuda:0")))
        nsm_loss = loss_func(target, pred).sum()
        loss_sum+=nsm_loss
        print('nsm_bias: {:.6f}'.format(nsm_loss.data))

    print("avg:"+str(float(loss_sum/len(nsm_data))))

    # single test
    pred, target = predict(net, input_data[-10:], label_data[-10:])
    pred = Variable(torch.Tensor(pred.cpu().detach().numpy() * output_std + output_mean).type(torch.FloatTensor).to(torch.device("cuda:0")))
    loss = loss_func(pred, target).sum()
    print('my_bias: {:.6f}'.format(loss.data))
