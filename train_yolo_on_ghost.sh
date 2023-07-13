python externals/yolov7/train.py --workers 8 --batch-size 4 \
    --data data/ghost_ship_01/data.yaml --img 1838 3270 --cfg cfg/training/yolov7-ghost.yaml \
    --weights '/home/carlyn.1/object_detection/runs/train/yolov7-rust63/weights/best.pt' \
    --name yolov7-ghost_01 --hyp data/hyp.scratch.ghost.yaml