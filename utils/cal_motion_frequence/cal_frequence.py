import pandas as pd
import argparse
import os
import motion_type as mt


def extract_files(date_path):
    file_num = 0
    all_motions = list()
    file_name_list = os.listdir(date_path)
    for file in file_name_list:
        if os.path.isdir(os.path.join(date_path, file)):
            sub_all_motions, sub_file_num = extract_files(os.path.join(date_path, file))
            file_num += sub_file_num
            all_motions += sub_all_motions
            continue

        # not marked
        if file[-7:-4] not in ["one", "two"]:
            continue

        # [motion_type, up_body_state, hands_state]
        motion_seq, up_seq, hands_seq = file[:-4].split('_')[-3:]
        motions = motion_seq.split("2")

        file_num += 1
        all_motions = all_motions + motions

    return all_motions, file_num


def _cal_frequence(ori_root, obj_root):
    df = pd.DataFrame()
    motion_type = mt.types["motion_type"]

    date_list = os.listdir(ori_root)
    date_list.sort(key=lambda x: int(x[:-4]))
    for date in date_list:
        date_path = os.path.join(ori_root, date)
        motion_dict = dict.fromkeys(motion_type, 0)
        all_motions, file_num = extract_files(date_path)

        # calc motion num
        for key in motion_dict:
            motion_dict[key] = all_motions.count(key)
        if file_num:
            df_date = pd.DataFrame([motion_dict])
            df_date.insert(0, "file_num", file_num)
            df_date.insert(0, "date", date)
            df = pd.concat([df, df_date])

    # calc sum
    col_sum = [int(item) for item in df.sum().to_list()]
    df.loc[len(df)] = col_sum
    df.iloc[-1, 0] = "all_dates"

    df.to_csv(obj_root, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_root", type=str, help="marked data")
    parser.add_argument("--obj_root", type=str, help="save the result")
    args = parser.parse_args()
    # 'E:\Locomotion'
    _cal_frequence(args.ori_root, args.obj_root)
