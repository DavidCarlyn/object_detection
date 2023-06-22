import random
import os
from enum import IntEnum
from argparse import ArgumentParser

from PIL import Image, ImageDraw

from utils import save_text

"""
Save dataset in COCO format for YOLO according to: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

class ToyLabels(IntEnum):
    SQUARE = 0
    TRIANGLE = 1
    CIRCLE = 2

    @classmethod
    def int_to_name(self, x):
        if x == 0: return "square"
        if x == 1: return "triangle"
        if x == 2: return "circle"

        return None

def draw_square(im, bb, color="green"):
    """
    im: image to draw on
    bb: [x0, y0, x1, y1] bounding box of rectangle
    color: string representing the color i.e. 'green'
    """
    d = ImageDraw.Draw(im)
    d.rectangle(bb, fill=color)

def draw_triangle(im, bb, color="green"):
    """
    im: image to draw on
    bb: [x0, y0, x1, y1] bounding box of triangle
    color: string representing the color i.e. 'green'
    """
    d = ImageDraw.Draw(im)
    x_mid = bb[2] - (bb[2] - bb[0])//2
    triangle_points = [(bb[0], bb[3]), (x_mid, bb[1]), (bb[2], bb[3])]
    d.polygon(triangle_points, fill=color)

def draw_circle(im, bb, color="green"):
    """
    im: image to draw on
    bb: [x0, y0, x1, y1] bounding box of triangle
    color: string representing the color i.e. 'green'
    """
    d = ImageDraw.Draw(im)
    circle_bb = [(bb[0], bb[1]), (bb[2], bb[3])]
    d.ellipse(circle_bb, fill=color)

def create_toy_image(imgs, positions, size=(64, 64)):
    toy_image = Image.new("RGB", size, color="black")
    for img, pos in zip(imgs, positions):
        toy_image.paste(img, pos)

    return toy_image

def draw_bbs(img, bbs, color="red"):
    rv = img.copy()

    d = ImageDraw.Draw(rv)
    for bb in bbs:
        d.rectangle(bb, outline=color)

    return rv

def create_toy_dataset(num=200, obj_range=(1, 4), obj_size_range=(4, 32), colors=["red", "green", "blue", "yellow", "orange", "purple"], img_size=(64,64)):
    """
    num: # of images to create for the dataset
    obj_range: 2-Tuple (min, max) of # of object a single image can have
    obj_size_range: 2-Tuple (min, max) of size an object in an image can have
    colors: list of colors an object can have
    img_size: the size of the images in the dataset
    """

    data = []
    for i in range(num):
        num_objs = random.randint(obj_range[0], obj_range[1])
        bbs = []
        lbls = []
        toy_image = Image.new("RGB", img_size, color="black")
        for j in range(num_objs):
            lbl = random.choice(list(ToyLabels))
            size = random.randint(obj_size_range[0], obj_size_range[1])
            color = random.choice(colors)

            x1 = random.randint(0, img_size[0] - size)
            y1 = random.randint(0, img_size[1] - size)
            x2 = x1 + (size-1)
            y2 = y1 + (size-1)
            bb = (x1, y1, x2, y2)


            if lbl == ToyLabels.SQUARE:
                draw_square(toy_image, bb, color)
            elif lbl == ToyLabels.TRIANGLE:
                draw_triangle(toy_image, bb, color)
            elif lbl == ToyLabels.CIRCLE:
                draw_circle(toy_image, bb, color)

            bbs.append(bb)
            lbls.append(lbl)
        
        data.append((toy_image, bbs, lbls))

    return data

def save_dataset(dataset, dir=os.path.join("data", "toy"), set_type="train"):
    """
    ../datasets/coco128/images/im0.jpg  # image
    ../datasets/coco128/labels/im0.txt  # label

    
    One row per object
    Each row is class x_center y_center width height format.
    Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
    Class numbers are zero-indexed (start from 0).

    """

    for i, (img, bbs, lbls) in enumerate(dataset):
        img_path = os.path.join(dir, set_type, "images", f"{i+1}.png")
        lbl_path = os.path.join(dir, set_type, "labels", f"{i+1}.txt")

        labels = []
        for bb, lbl in zip(bbs, lbls):
            box_w, box_h = bb[2]-bb[0]+1, bb[3]-bb[1]+1
            box_x, box_y = bb[0] + (box_w / 2), bb[1] + (box_h / 2)

            box_x, box_y, box_w, box_h = box_x / img.width, box_y / img.height, box_w / img.width, box_h / img.height

            labels.append(f"{lbl} {box_x} {box_y} {box_w} {box_h}\n")

        img.save(img_path)
        save_text(labels, lbl_path, is_list=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_imgs", type=int, default=200)
    parser.add_argument("--val_imgs", type=int, default=200)
    parser.add_argument("--test_imgs", type=int, default=200)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    dataset = create_toy_dataset(args.train_imgs, img_size=(args.img_size, args.img_size))
    os.makedirs(os.path.join("data", "toy", "train", "images"), exist_ok=True)
    os.makedirs(os.path.join("data", "toy", "train", "labels"), exist_ok=True)
    save_dataset(dataset, set_type="train")
    dataset = create_toy_dataset(args.val_imgs, img_size=(args.img_size, args.img_size))
    os.makedirs(os.path.join("data", "toy", "val", "images"), exist_ok=True)
    os.makedirs(os.path.join("data", "toy", "val", "labels"), exist_ok=True)
    save_dataset(dataset, set_type="val")
    dataset = create_toy_dataset(args.test_imgs, img_size=(args.img_size, args.img_size))
    os.makedirs(os.path.join("data", "toy", "test", "images"), exist_ok=True)
    os.makedirs(os.path.join("data", "toy", "test", "labels"), exist_ok=True)
    save_dataset(dataset, set_type="test")