python ./externals/yolov7_seg/seg/segment/train.py --workers 8 --batch-size 16 \
    --weights '/home/carlyn.1/object_detection/externals/yolov7_seg/seg/runs/train-seg/yolov7-our_corrosion_seg/weights/best.pt' \
    --data /home/carlyn.1/object_detection/data/our_corrosion/corrosion.yaml \
    --img 640 --cfg ./externals/yolov7/cfg/training/yolov7-corrosion.yaml --name yolov7-our_corrosion_seg \
    --hyp ./externals/yolov7/data/hyp.scratch.corrosion.yaml