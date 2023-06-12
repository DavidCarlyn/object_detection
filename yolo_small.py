import sys
import torch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T

from datasets import ToyDetectionDataset

class SmallConvNet(nn.Module):
    def __init__(self):
        pass

def get_transforms():
    return T.Compose([
        T.ToTensor()
    ])

if __name__ == "__main__":
    dataset = ToyDetectionDataset("data/toy", transforms=get_transforms())
    item = dataset[0]
    print(item[0])
    print(item[1])
    print(item[2])