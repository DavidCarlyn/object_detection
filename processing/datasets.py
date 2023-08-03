import os

import torch
from PIL import Image

from utils import load_json

class ToyDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, set_type="train", transforms=None):
        self.transforms = transforms
        self.img_paths = []
        for root_dir, dirs, paths in os.walk(os.join(root, set_type, "images")):
            for path in paths:
                ext = os.path.splitext(path)[1]
                if ext != ".png": continue
                self.img_paths.append(os.path.join(root, path))

        self.annotations = load_json(os.path.join(root, set_type, "labels.json"))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        annotations = self.annotations[idx]

        bboxs = torch.as_tensor(annotations["bbs"], dtype=torch.float32)
        lbls = torch.as_tensor(annotations["lbls"], dtype=torch.int64)

        if self.transforms is not None:
            #img, bboxs = self.transforms(img, bboxs)
            img = self.transforms(img)

        return img, bboxs, lbls

