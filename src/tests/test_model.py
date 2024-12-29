import unittest

import torch

from classify.model import ResNet50


class TestModel(unittest.TestCase):
    def test_model(self):
        # 创建ResNet50模型实例
        model = ResNet50()
        # 随机生成一个输入张量，模拟输入图像数据
        input_tensor = torch.randn(1, 3, 224, 224)
        # 将输入张量传入模型，得到输出
        output = model(input_tensor)
        print(output.shape)