import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_fig(data, save_path):
    color1 = "y*-"
    color2 = "r--"
    plt.plot(data.index, data, color1)
    plt.savefig(save_path)
    plt.show()
    plt.close


if __name__ == '__main__':

    my_path = "../Server/test.csv"
    my_output = pd.read_csv(my_path, header=None, dtype='float')
    my_avg = np.mean(my_output, axis=0)
    my_std = np.std(my_output, axis=0)

    # nsm_path = "/home/rr/Downloads/nsm_data/nsm_data.xlsx"
    # nsm_output = pd.read_excel(nsm_path, sheet_name="Output")
    nsm_path = "/home/rr/Downloads/nsm_data/nsm_walk_data/Output.txt"
    nsm_output = pd.read_csv(nsm_path, sep=' ', header=None, dtype=float)
    nsm_output = nsm_output.iloc[:, 0:618]
    nsm_avg = np.mean(nsm_output, axis=0)
    nsm_std = np.std(nsm_output, axis=0)

    # save_fig(my_avg, "walk_compare_res/my_avg.png")
    # save_fig(nsm_avg, "walk_compare_res/nsm_avg.png")
    # save_fig(abs(my_avg-nsm_avg), "walk_compare_res/abs_dis_avg.png")

    label_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/OutputLabels.txt"
    labels = pd.read_csv(label_path, header=None, sep=' ')
    labels["small_flag"] = 0

    # 寻找差值
    dis = abs(my_avg-nsm_avg)
    small_err, small_err_index = [], []
    big_err, big_err_index = [], []
    for index, err in enumerate(dis):
        if err < 1.0:
            small_err.append(round(err, 5))
            small_err_index.append(index)
        else:
            big_err.append(round(err, 5))
            big_err_index.append(index)

    names = np.array(labels.iloc[big_err_index, 1])
    merge_list = [item for item in zip(big_err_index, big_err, names)]
    merge_list = pd.DataFrame(merge_list)
    merge_list.to_csv("walk_compare_res/err>10.csv", header=None, index=None)

    # 差值
    # dis = abs(my_avg-nsm_avg)
    # small_err, small_err_index = [], []
    # big_err, big_err_index = [], []
    # small_one_hot = [1 if i < 0.1 else 0 for i in dis]
    # for index, err in enumerate(dis):
    #     if err < 0.1:
    #         labels.iloc[index, -1] = 1
    #         small_err.append(err)
    #         small_err_index.append(index)
    #     else:
    #         big_err.append(err)
    #         big_err_index.append(index)

    # labels.iloc[small_err_index].to_csv("walk_compare_res/"+str(len(small_err_index))+"small_err_item_list.csv", header=None, index=None)
    # labels.iloc[big_err_index].to_csv("walk_compare_res/"+str(len(big_err_index))+"big_err_item_list.csv", header=None, index=None)
    # labels.to_csv("walk_compare_res/"+str(len(small_err_index))+"s_mixed_err_item_list", index=None)

    # 绘制差值分布直方图
    # bins = np.linspace(min(dis), max(dis), 50)
    # plt.hist(abs(my_avg-nsm_avg), bins)
    # plt.title('Dis Frequency distribution')
    # plt.savefig("walk_compare_res/dis_freq_distribution.png")
    # plt.show()
    # plt.close()

    # left = 200
    # right = 300
    # plt.plot(range(len(small_one_hot[left:right])), small_one_hot[left:right], 'r.')
    # plt.show()
