python /home/carlyn.1/object_detection/externals/yolov7_seg/seg/segment/train.py --workers 8 --batch-size 1 \
    --data /home/carlyn.1/object_detection/data/ghost_ship_segmentation/data.yaml --img 1856 \
    --cfg ./configs/model/yolov7-ghost_seg.yaml --weights /home/carlyn.1/object_detection/data/weights/yolov7-seg.pt \
    --name yolov7-ghost_seg --hyp ./configs/training/hyp.scratch.ghost_seg.yaml