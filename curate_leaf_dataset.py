import os
import random
from enum import IntEnum
from argparse import ArgumentParser

import xml.etree.ElementTree as ET
from tqdm import tqdm

from PIL import Image

from utils import save_text

class LeafLabels(IntEnum):
    BICHO_MINEIRO = 0
    FERRUGEM = 1

    @classmethod
    def int_to_name(self, x):
        if x == 0: return "bicho_mineiro"
        if x == 1: return "ferrugem"

        return None
    
def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    img_data = {}
    size_xml = root.find('size')
    width = int(size_xml.find('width').text)
    height = int(size_xml.find('height').text)
    channels = int(size_xml.find('depth').text)

    bbs = []
    for obj in root.findall('object'):
        bb_xml = obj.find('bndbox')
        x0 = int(bb_xml.find('xmin').text)
        y0 = int(bb_xml.find('ymin').text)
        x1 = int(bb_xml.find('xmax').text)
        y1 = int(bb_xml.find('ymax').text)

        box_w, box_h = x1-x0+1, y1-y0+1
        box_x, box_y = x0 + (box_w / 2), y0 + (box_h / 2)

        box_x, box_y, box_w, box_h = box_x / width, box_y / height, box_w / width, box_h / height
        bbs.append([box_x, box_y, box_w, box_h])
    return bbs

def extract_data(dir_path, lbl):
    basepaths = set()
    for root, _, files in os.walk(dir_path):
        for fname in files:
            name, ext = os.path.splitext(fname)
            basepaths.add(os.path.join(root, name))

    data = []
    for path in basepaths:
        bbs = parse_xml(path + ".xml")
        data.append([lbl, path + ".jpg", bbs])

    return data

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--mineiro_path", type=str, default="data/miner_img_xml")
    parser.add_argument("--rust_path", type=str, default="data/rust_xml_image")
    parser.add_argument("--splits", nargs='+', type=float, default=[0.7, 0.1]) # Test is inferred

    args = parser.parse_args()

    assert len(args.splits) == 2, "Must provide train and validation splits for --splits"

    return args

def save_data(data, root_path):
    for i, (lbl, img_path, bbs) in tqdm(enumerate(data)):
        # Save Image
        Image.open(img_path).save(os.path.join(root_path, "images", f"{i+1}.png"))

        # Save Annotations
        anno = []
        for bb in bbs:
            anno.append(str(lbl) + " ".join(map(lambda x: str(x), bb)) + "\n")
        save_text(anno, os.path.join(root_path, "labels", f"{i+1}.txt"), is_list=True)

if __name__ == "__main__":
    args = get_args()

    # Extract data
    mineiro_data = extract_data(args.mineiro_path, LeafLabels.BICHO_MINEIRO)
    rust_data = extract_data(args.rust_path, LeafLabels.FERRUGEM)
    combined_data = mineiro_data + rust_data
    
    # Shuffle and split
    random.shuffle(combined_data)

    # Make dirs
    for phase in ["train", "val", "test"]:
        for type in ["images", "labels"]:
            os.makedirs(os.path.join("data", "leaf", phase, type), exist_ok=True)

    # Train
    t0 = int(len(combined_data) * args.splits[0])
    train_data = combined_data[:t0]
    save_data(train_data, os.path.join("data", "leaf", "train"))

    # Val
    t1 = int(len(combined_data) * sum(args.splits))
    val_data = combined_data[t0:t1]
    save_data(val_data, os.path.join("data", "leaf", "val"))
    
    # Test
    test_data = combined_data[t1:]
    save_data(test_data, os.path.join("data", "leaf", "test"))

