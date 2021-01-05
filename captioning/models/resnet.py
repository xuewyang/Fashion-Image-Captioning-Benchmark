import torch
from torch import nn
import torchvision


class ResNet(nn.Module):
    """
    The ResNet we have to finetune.
    """
    def __init__(self, num_cate):
        """
        :param num_cate: number of categories
        """
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        self.resnet.fc = nn.Linear(2048, num_cate)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: logits
        """
        logits = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        return logits