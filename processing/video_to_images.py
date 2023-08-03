import os
from argparse import ArgumentParser

import cv2

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    vidcap = cv2.VideoCapture(args.video)
    success, img = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(f"{args.save_dir}/ghost_ship_{count}.jpg", img)
        print(f"Saved image: {count}")
        success, img = vidcap.read()
        count += 1
