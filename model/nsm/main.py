import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from utils.utils import get_norm
from torch.autograd import Variable
from net import MyModel
from my_config import my_conf


def train(net, inputs_list):
    f = open("model/record.txt", "w")
    input_mean, input_std = get_norm("C:/Users/rr/Desktop/documents/Export/InputNorm.txt")
    output_mean, output_std = get_norm("C:/Users/rr/Desktop/documents/Export/OutputNorm.txt")

    for epoch in range(net.epoch):
        train_input_data = pd.DataFrame()
        train_label_data = pd.DataFrame()
        all_test_loss = []
        train_loss_sum, n = 0.0, 0
        files_num = 50  # samll-10  all-50

        for i, file in enumerate(tqdm(inputs_list)):
            if i % files_num:
                single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
                single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
                train_input_data = train_input_data.append(single_input_data, ignore_index=True)
                train_label_data = train_label_data.append(single_label_data, ignore_index=True)
            elif i != 0 and i % files_num == 0:
                # 标准化
                t_input_data = torch.Tensor((np.array(train_input_data) - input_mean) / input_std)
                t_label_data = torch.Tensor((np.array(train_label_data) - output_mean) / output_std)
                # t_input_data = torch.Tensor(np.array(train_input_data))
                # t_label_data = torch.Tensor(np.array(train_label_data))
                t_input_data = Variable(t_input_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
                t_label_data = Variable(t_label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

                train_size = int(len(t_input_data) * 0.7 // net.batch_size * net.batch_size)
                for t_i in range(0, train_size, net.batch_size):
                    x = t_input_data[t_i:t_i + net.batch_size, :]
                    y = t_label_data[t_i:t_i + net.batch_size, :]

                    y_hat = net.forward(x)
                    loss = net.loss_function(y_hat, y).sum()
                    # print(loss)
                    net.optimizer.zero_grad()

                    loss.backward()
                    net.optimizer.step()

                    train_loss_sum += loss.item()
                    n += 1

                test_loss = test(net, t_input_data, t_label_data, train_size, net.loss_function, net.batch_size)
                all_test_loss = all_test_loss + [float(test_loss)]
                train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
                train_label_data = train_label_data.drop(train_label_data.index, inplace=False)

        print('epoch %d, loss %.7f. test_loss %.7f' % (epoch, train_loss_sum / n, np.mean(all_test_loss)))
        item = str(epoch) + ' ' + str(train_loss_sum / n) + ' ' + str(np.mean(all_test_loss)) + '\n'
        f.write(item)
        f.flush()

        if epoch % 50 == 0 and epoch != 0:
            for i in range(net.encoder_nums):
                net.encoders[i].module.save_network(i, net.save_path)
            net.gating.module.save_network(-1, net.save_path)
            for i in range(net.expert_nums):
                net.experts[i].module.save_network(i, net.save_path)
            # save model for load weights
            for i in range(net.encoder_nums):
                torch.save(net.encoders[i].state_dict(), os.path.join("model/", str(epoch) + 'encoder%0i.pth' % i))
            torch.save(net.gating.state_dict(), os.path.join(net.save_path, str(epoch) + 'gating.pth'))
            for i in range(net.expert_nums):
                torch.save(net.experts[i].state_dict(), os.path.join("model/", str(epoch) + 'expert%0i.pth' % i))
            torch.save(net.optimizer.state_dict(), os.path.join("model/", 'optimizer' + str(epoch) + '.ptm'))
    f.close()


def test(net, input_data, label_data, train_size, loss_func, batch_size):
    test_loss_sum, n = 0.0, 0
    length = len(input_data) // batch_size * batch_size
    for t_i in range(train_size, length, batch_size):
        X = input_data[t_i:t_i + batch_size, :]
        y = label_data[t_i:t_i + batch_size, :]
        y_hat = net.forward(X)

        loss = loss_func(y_hat, y).sum()
        test_loss_sum += loss
        n += 1
    return test_loss_sum / n


if __name__ == '__main__':
    root_path = "D:/nsm_data/Train/"
    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    net = MyModel(**my_conf["model"])
    train(net, inputs_list)
