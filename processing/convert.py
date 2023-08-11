from argparse import ArgumentParser

from globox import AnnotationSet

def convert(src_format, dest_format, src_path, src_path2, dest_path):
    print(src_format, dest_format)
    if src_format == "yolo":
        src_anno = AnnotationSet.from_yolo_v5(
            folder = src_path,
            image_folder = src_path2
        )
    elif src_format == "coco":
        src_anno = AnnotationSet.from_coco(
            file_path = src_path
        )
    elif src_format == "cvat":
        src_anno = AnnotationSet.from_cvat(
            file_path = src_path,
            verbose=True)

    if dest_format == "yolo":
        src_anno.save_yolo_v5(
            save_dir=dest_path, 
            label_to_id={"rust": 0},
            verbose=True)
    elif dest_format == "coco":
        src_anno.save_coco(
            path=dest_path, 
            label_to_id={"rust": 0},
            auto_ids=True)
    elif dest_format == "cvat":
        src_anno.save_cvat(dest_path,verbose=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_format", type=str, choices=["yolo", "coco", "cvat"], default="coco")
    parser.add_argument("--dest_format", type=str, choices=["yolo", "coco", "cvat"], default="yolo")
    parser.add_argument("--src_path", type=str, default=None)
    parser.add_argument("--src_path2", type=str, default=None)
    parser.add_argument("--dest_path", type=str, default=None)
    args = parser.parse_args()

    convert(args.src_format, args.dest_format, args.src_path, args.src_path2, args.dest_path)
    