import unittest

import pandas


class TestPandas(unittest.TestCase):
    def test_pandas(self):
        fp = r'D:\py\engineering\classify-leaves\resources\test.xlsx'
        df=pandas.read_excel(fp, header=0, index_col=None, usecols=[0, 1], skiprows=1)
        print('\n', df.head())
