from argparse import ArgumentParser

from PIL import Image, ImageDraw

import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--lbl", type=str, default=None)
    args = parser.parse_args()
    
    assert args.img is not None and args.lbl is not None, "provide path to image and label"

    bbs = []
    with open(args.lbl, 'r') as f:
        lines = f.readlines()
        for line in lines:
            bbs.append(line.strip().split(" ")[1:])
    
    img = Image.open(args.img)
    np_img = np.array(img)
    height, width, _ = np_img.shape

    draw = ImageDraw.Draw(img)

    for (x, y, w, h) in bbs:
        #print(x, y, w, h, width, height)
        x = int(float(x) * width)
        y = int(float(y) * height)
        w = int(float(w) * width)
        h = int(float(h) * height)

        x0 = x - w//2
        y0 = y - h//2
        x1 = x + w//2
        y1 = y + h//2

        draw.rectangle([x0, y0, x1, y1])

    img.save("gt.png")

