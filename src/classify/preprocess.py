import os
import pandas as pd
import math

import numpy as np


def get_tmp_label_file(root_dir, src, 
                       dests=["tmp_train.csv", "tmp_val.csv", "tmp_test.csv", "label.txt"], 
                       split=[0.7, 0.2, 0.1]):
    r"""将原始标签文件进行预处理
    """
    data = pd.read_csv(os.path.join(root_dir, src))
    # 将DataFrame转换为numpy数组
    data_array = data.values
    data_len = len(data_array)
    np.random.shuffle(data_array)
    train_idx = math.floor(data_len * split[0])
    val_idx = math.floor(data_len * (split[0] + split[1]))
    # 划分数据集
    shuffled_train_data = pd.DataFrame(data_array[0:train_idx], columns = data.columns)
    shuffled_train_data.index = range(len(shuffled_train_data))
    shuffled_train_data.to_csv(os.path.join(root_dir, dests[0]), index=False)

    shuffled_val_data = pd.DataFrame(data_array[train_idx + 1:val_idx], columns = data.columns)
    shuffled_val_data.index = range(len(shuffled_val_data))
    shuffled_val_data.to_csv(os.path.join(root_dir, dests[1]), index=False)

    shuffled_test_data = pd.DataFrame(data_array[val_idx:], columns = data.columns)
    shuffled_test_data.index = range(len(shuffled_test_data))
    shuffled_test_data.to_csv(os.path.join(root_dir, dests[2]), index=False)

    # 获取类别数据
    shuffled_data = pd.DataFrame(data_array, columns = data.columns)
    labels = shuffled_data['label'].drop_duplicates().values.tolist()
    with open(os.path.join(root_dir, dests[3]), 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{label}\n")


if __name__ == "__main__":
    src_dir = r"/home/dcd/zww/data/classify-leaves"
    src_file = "train.csv"
    get_tmp_label_file(src_dir, src_file)
