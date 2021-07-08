import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from utils.utils import get_norm
from net import Model
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


def train(net, inputs_list, num_epochs):
    f = open("model_list/part_BN_outscale_3l_4h_models/plots/noBN_3l_4h.txt", "w")
    input_mean, input_std = get_norm("data/InputNorm.txt")
    input_mean, input_std = input_mean[0:926], input_std[0:926]
    output_mean, output_std = get_norm("data/OutputNorm.txt")

    for epoch in range(num_epochs):
        train_input_data = pd.DataFrame()
        train_label_data = pd.DataFrame()
        scale = StandardScaler()
        all_test_loss = []
        train_loss_sum, n = 0.0, 0
        files_num = 10  # samll-10  all-50
        train_size = int(files_num * 1000 * 0.7)

        for i, file in enumerate(tqdm(inputs_list)):
            if i % files_num:
                single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
                single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
                train_input_data = train_input_data.append(single_input_data, ignore_index=True)
                train_label_data = train_label_data.append(single_label_data, ignore_index=True)
            elif i != 0 and i % files_num == 0:
                # scale 标准化
                # scale = scale.fit(train_input_data)
                # t_input_data = torch.Tensor(scale.transform(train_input_data))

                # 标准化
                t_input_data = torch.Tensor((np.array(train_input_data) - input_mean) / input_std)
                t_label_data = torch.Tensor((np.array(train_label_data) - output_mean) / output_std)

                # t_input_data = torch.Tensor(np.array(train_input_data))
                # t_label_data = torch.Tensor(np.array(train_label_data))

                t_input_data = Variable(t_input_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
                t_label_data = Variable(t_label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

                for t_i in range(0, train_size, net.batch_size):
                    X = t_input_data[t_i:t_i + net.batch_size, :]
                    y = t_label_data[t_i:t_i + net.batch_size, :]
                    y_hat = net.model(X)

                    loss = net.loss_func(y_hat, y).sum()
                    # print(loss)
                    net.optimizer.zero_grad()

                    loss.backward()
                    net.optimizer.step()

                    train_loss_sum += loss.item()
                    n += 1

                test_loss = test(net, t_input_data, t_label_data, train_size, net.loss_func, net.batch_size)
                all_test_loss = all_test_loss + [float(test_loss)]
                train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
                train_label_data = train_label_data.drop(train_label_data.index, inplace=False)

        print('epoch %d, loss %.7f. test_loss %.7f' % (epoch, train_loss_sum / n, np.mean(all_test_loss)))
        item = str(epoch) + ' ' + str(train_loss_sum / n) + ' ' + str(np.mean(all_test_loss)) + '\n'
        f.write(item)
        f.flush()

        if epoch % 4 == 0:
            torch.save(net.model.state_dict(), os.path.join("model_list/part_BN_outscale_3l_4h_models/", "fcn_" + str(epoch) + ".pth"))
            torch.save(net.optimizer.state_dict(), os.path.join("model_list/part_BN_outscale_3l_4h_models/", "fcn_opt_" + str(epoch) + ".pth"))
    f.close()


def test(net, input_data, label_data, train_size, loss_func, batch_size):
    test_loss_sum, n = 0.0, 0
    for t_i in range(train_size, len(input_data), batch_size):
        X = input_data[t_i:t_i + batch_size, :]
        y = label_data[t_i:t_i + batch_size, :]
        y_hat = net.model(X)

        loss = loss_func(y_hat, y).sum()
        test_loss_sum += loss
        n += 1

    return test_loss_sum / n


if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 926, 618, 926*4    #926,618,256
    learning_rate, batch_size, num_epochs = 0.1, 64, 600
    # root_path = "D:/nsm_data/bone_gating_WalkTrain/"
    root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"

    # num_inputs, num_outputs, num_hiddens = 5307, 618, 1024
    # learning_rate, batch_size, num_epochs = 0.01, 32, 200
    # root_path = "/home/rr/Downloads/nsm_data/Train/"

    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    net = Model(num_inputs, num_hiddens, num_outputs, learning_rate, batch_size)
    train(net, inputs_list, num_epochs)
