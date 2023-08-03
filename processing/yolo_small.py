import sys
import torch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T

from datasets import ToyDetectionDataset

class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return out

def get_transforms():
    return T.Compose([
        T.ToTensor()
    ])

if __name__ == "__main__":
    dataset = ToyDetectionDataset("data/toy", transforms=get_transforms())
    item = dataset[0]
    
    feature_extractor = SmallConvNet()