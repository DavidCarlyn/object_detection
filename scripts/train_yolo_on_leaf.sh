python externals/yolov7/train.py --workers 8 --batch-size 32 --data data/leaf/leaf.yaml --img 512 512 --cfg cfg/training/yolov7-leaf.yaml --weights 'data/weights/yolov7_training.pt' --name yolov7-leaf --hyp data/hyp.scratch.leaf.yaml