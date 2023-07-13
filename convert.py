from argparse import ArgumentParser

from globox import AnnotationSet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_format", type=str, choices=["yolo", "coco"], default="coco")
    parser.add_argument("--dest_format", type=str, choices=["yolo", "coco", "cvat"], default="yolo")
    parser.add_argument("--src_path", type=str, default=None)
    parser.add_argument("--src_path2", type=str, default=None)
    parser.add_argument("--dest_path", type=str, default=None)
    args = parser.parse_args()

    if args.src_format == "yolo":
        src_anno = AnnotationSet.from_yolo_v5(
            folder = args.src_path,
            image_folder = args.src_path2
        )
    elif args.src_format == "coco":
        src_anno = AnnotationSet.from_coco(
            file_path = args.src_path
        )

    if args.dest_format == "yolo":
        src_anno.save_yolo_v5(
            save_dir=args.dest_path, 
            label_to_id={"rust": 0})
    elif args.dest_format == "coco":
        src_anno.save_coco(
            path=args.dest_path, 
            label_to_id={"rust": 0},
            auto_ids=True)
    elif args.dest_format == "cvat":
        src_anno.save_cvat(args.dest_path)
    