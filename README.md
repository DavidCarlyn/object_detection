# object_detection
Playground for learning ML object detection techniques

# Adding to this Repo:

If adding a new dependency, please update environment.yaml file with the following command:

```
pip list --format=freeze > requirements.txt
```

# Installation

Be sure to clone the repository with the submodules

```
!git clone --recurse-submodules https://github.com/DavidCarlyn/object_detection.git
```

Install dependencies

```
pip install -r requirements.txt
```

# Usage

See scripts for example usage. Data will have to be configured to YOLO format. Configuration can be found in either of these repositories: 

https://github.com/WongKinYiu/yolov7
https://github.com/RizwanMunawar/yolov7-segmentation
https://github.com/ultralytics/yolov5
