python externals/yolov7/train.py --workers 8 --batch-size 4 \
    --data data/ghost_ship_01/data.yaml --img 1838 --cfg configs/model/yolov7-ghost.yaml \
    --weights '/home/carlyn.1/object_detection/runs/train/yolov7-rust63/weights/best.pt' \
    --name yolov7-ghost_01 --hyp /configs/training/hyp.scratch.ghost.yaml