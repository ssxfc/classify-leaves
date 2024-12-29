import unittest

import pandas


class TestPandas(unittest.TestCase):
    def test_pandas(self):
        fp = r'D:\py\engineering\classify-leaves\resources\train.csv'
        data = pandas.read_csv(fp)
        df_train = pandas.DataFrame(data)
        df_train = df_train.sort_values(by="label")
        print(len(df_train))

        # zip可迭代
        for x in zip(df_train['image'], df_train['label']):
            print(x)

        # 遍历列的可能值
        for label in df_train['label'].drop_duplicates():
            print(label)

        # 按列值选取样本
        t = df_train['label'] == 'tilia_cordata'
        print(t)
        imgs = df_train[t]['image'].to_list()
        print(imgs)

        # 存储csv
        new_csv_list = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        require_saving = pandas.DataFrame(new_csv_list, columns=['first_col', 'second_col'])
        require_saving.to_csv(r'D:\py\engineering\classify-leaves\resources\demo.csv', index=False)
