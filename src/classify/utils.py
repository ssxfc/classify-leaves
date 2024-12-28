import os
import shutil
import pandas as pd
from math import *
from tqdm import tqdm

root_dir = r"D:\datasets\classify-leaves"


ori_img_path = os.path.join(root_dir, "images")
train_img_path = os.path.join(root_dir, "train_img")
vaild_img_path = os.path.join(root_dir, "vaild_img")
test_img_path = os.path.join(root_dir, "test_img", "unknown")


def preprocess_labelfiles():
    r"""将原始标签文件进行预处理
    """
    # train data
    data = pd.read_csv(os.path.join(root_dir, "train.csv"))
    df_train = pd.DataFrame(data)
    # label_encoder = LabelEncoder()
    # df_train["label"] = label_encoder.fit_transform(df_train["label"])
    df_train = df_train.sort_values(by="label")
    df_train["image"] = df_train["image"].apply(lambda x: x.split("/")[1].split(".")[0])
    df_train.to_csv(os.path.join(root_dir, "train_labeled.csv"), index=False)
    # test data
    data = pd.read_csv(os.path.join(root_dir, "test.csv"))
    df_test = pd.DataFrame(data)
    df_test["image"] = df_test["image"].apply(lambda x: x.split("/")[1].split(".")[0])
    df_test.to_csv(os.path.join(root_dir, "test_labeled.csv"), index=False)


def get_train_path(target):
    return os.path.join(train_img_path, str(target))


def get_filepath(target):
    return os.path.join(ori_img_path, f"{str(target)}.jpg")


def get_vaild_path(target):
    return os.path.join(vaild_img_path, str(target))


def get_vaild_filepath(file, label):
    return os.path.join(train_img_path, os.path.join(str(label), f"{str(file)}.jpg"))


def copyfile(filepath, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filepath, target_dir)


def movefile(filepath, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(filepath, target_dir)


def split_train_and_vaild(df: pd.DataFrame, ratio):
    # 把所有图像先存放在train_path中
    for img, label in tqdm(zip(df["image"], df["label"]), total=len(df)):
        copyfile(get_filepath(img), get_train_path(label))

    # 把valid需要的图像移动到valid_path中
    df_vaild = []
    for label in tqdm(df["label"].drop_duplicates(), total=176):
        img = df[df["label"] == label]["image"].to_list()
        # 按类别对数据进行 1-ratio : ratio 划分为训练集和验证集
        for i in img[0 : floor(len(img) * ratio)]:
            df_vaild.append([i, label])
            # 把被划分到验证集的数据从df中移除
            df.drop(df[df["image"] == i].index, inplace=True)
            movefile(get_vaild_filepath(i, label), get_vaild_path(label))
    # 更新标签文件
    df_vaild = pd.DataFrame(df_vaild, columns=["image", "label"])
    df_vaild.to_csv(os.path.join(root_dir, "vaild_splited.csv"), index=False)
    df.to_csv(os.path.join(root_dir, "train_splited.csv"), index=False)


def prepare_test(df: pd.DataFrame):
    r"""转存测试数据到临时的测试目录中
    """
    for img in tqdm(df["image"], total=len(df)):
        copyfile(get_filepath(img), os.path.join(test_img_path))


if __name__ == "__main__":
    preprocess_labelfiles()

    data = pd.read_csv(os.path.join(root_dir, "train_labeled.csv"))
    df_train = pd.DataFrame(data)
    split_train_and_vaild(df_train, 0.1)

    data = pd.read_csv(os.path.join(root_dir, "test_labeled.csv"))
    df_test = pd.DataFrame(data)
    prepare_test(df_test)
