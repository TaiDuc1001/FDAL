from dotenv import load_dotenv
load_dotenv()

import os
import json
import shutil
import argparse
from glob import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rich.progress import Progress
import xml.etree.ElementTree as ET
import pandas as pd
import yaml

from ultralytics.utils.downloads import download
from kaggle.api.kaggle_api_extended import KaggleApi

VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
    "parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]

def convert_label(path, lb_path, year, image_id, dataset_name):
    def convert_box(size, box):
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f"VOC{year}/Annotations/{image_id}.xml")
    out_file = open(lb_path, "w")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text) # type: ignore
    h = int(size.find("height").text) # type: ignore

    names = VOC_NAMES
    for obj in root.iter("object"):
        cls = obj.find("name").text # type: ignore
        if cls in names and int(obj.find("difficult").text) != 1: # type: ignore
            xmlbox = obj.find("bndbox")
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")]) # type: ignore
            cls_id = names.index(cls)
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + "\n")

def down_VOC(zipdir=None):
    # dir = Path("datasets/VOC")
    # url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    # urls = [
    #     f"{url}VOCtrainval_06-Nov-2007.zip",  # 446MB, 5012 images
    #     f"{url}VOCtest_06-Nov-2007.zip",  # 438MB, 4953 images
    #     f"{url}VOCtrainval_11-May-2012.zip",  # 1.95GB, 17126 images
    # ]
    # try:
    #     download(urls, dir=dir / "images", curl=True, threads=3, exist_ok=True)
    # except FileExistsError:
    #     download(urls, dir=dir / "images", curl=True, threads=1, exist_ok=True)
    # path = dir / "images/VOCdevkit"
    # for year, image_set in ("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val"), ("2007", "test"):
    #     imgs_path = dir / "images" / f"{image_set}{year}"
    #     lbs_path = dir / "labels" / f"{image_set}{year}"
    #     imgs_path.mkdir(exist_ok=True, parents=True)
    #     lbs_path.mkdir(exist_ok=True, parents=True)

    #     with open(path / f"VOC{year}/ImageSets/Main/{image_set}.txt") as f:
    #         image_ids = f.read().strip().split()
        
    #     with Progress() as progress:
    #         task = progress.add_task(f"[cyan]{image_set}{year}", total=len(image_ids))
    #         for id in image_ids:
    #             f = path / f"VOC{year}/JPEGImages/{id}.jpg"
    #             lb_path = (lbs_path / f.name).with_suffix(".txt")
    #             f.rename(imgs_path / f.name)
    #             convert_label(path, lb_path, year, id, "VOC")
    #             progress.update(task, advance=1)
    # for zip_file in (dir / "images").glob("*.zip"):
    #     zip_file.unlink()
    voc_dir = Path("datasets")
    voc_dir.mkdir(exist_ok=True, parents=True)
    dataset_id = 'taiducphan/voc0712-yolo'
    api = KaggleApi()
    api.authenticate()
    if zipdir:
        zipdir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=zipdir.as_posix(), unzip=True)
        shutil.move(str(zipdir / "VOC"), str(voc_dir / "VOC"))
    else:
        api.dataset_download_files(dataset_id, path=voc_dir.as_posix(), unzip=True)
    voc_dir = Path("datasets/VOC")
    yaml_path = voc_dir / "data.yaml"
    data_yaml = []
    data_yaml.append(f"path: {voc_dir}")
    data_yaml.append("train:")
    data_yaml.append(f"  - {voc_dir}/images/train2012")
    data_yaml.append(f"  - {voc_dir}/images/train2007")
    data_yaml.append(f"  - {voc_dir}/images/val2012")
    data_yaml.append(f"  - {voc_dir}/images/val2007")
    data_yaml.append(f"val: {voc_dir}/images/test2007")
    data_yaml.append(f"nc: {len(VOC_NAMES)}")
    data_yaml.append("names:")
    for n in VOC_NAMES:
        data_yaml.append(f"  - {n}")
    yaml_path.write_text("\n".join(str(x) for x in data_yaml))
    print(f"data.yaml saved to: {yaml_path.resolve()}")

def down_COCO(zipdir=None):
    api = KaggleApi()
    api.authenticate()

    dataset_id = "sarkisshilgevorkyan/coco-dataset-for-yolo"
    download_dir = Path("datasets")
    download_dir.mkdir(parents=True, exist_ok=True)
    train_imgs = glob("datasets/COCO/images/train2017/*")
    if len(train_imgs) > 100000:
        print("COCO dataset already exists. Skipping download.")
    else:
        if zipdir:
            zipdir.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(dataset_id, path=zipdir.as_posix(), unzip=True)
            shutil.move(str(zipdir / "coco"), str(download_dir / "coco"))
        else:
            api.dataset_download_files(dataset_id, path=download_dir.as_posix(), unzip=True)
        os.system('mv datasets/coco datasets/COCO')
    download_dir = download_dir / "COCO"
    yaml_path = download_dir / "data.yaml"
    data_yaml = []
    data_yaml.append(f"path: {download_dir}")
    data_yaml.append("train: images/train2017")
    data_yaml.append("val: images/val2017")
    data_yaml.append(f"nc: {len(COCO_NAMES)}")
    data_yaml.append("names:")
    for n in COCO_NAMES:
        data_yaml.append(f"  - {n}")
    yaml_path.write_text("\n".join(str(x) for x in data_yaml))
    print(f"data.yaml saved to: {yaml_path.resolve()}")

    labels_root = download_dir / "labels"

    def process_label_file(path):
        text = path.read_text().strip().splitlines()
        out_lines = []
        changed = False
        for i, line in enumerate(text, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                out_lines.append(line)
                continue
            cls = parts[0]
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                out_lines.append(line)
                continue
            if len(vals) == 4:
                out_lines.append(f"{cls} {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f}")
                continue
            if len(vals) >= 6 and len(vals) % 2 == 0:
                xs = vals[0::2]
                ys = vals[1::2]
                max_x = max(xs)
                min_x = min(xs)
                max_y = max(ys)
                min_y = min(ys)
                if max_x > 1.0 or max_y > 1.0:
                    print(f"{path}: line {i} contains absolute coordinates (>1). Skipping normalization.")
                    out_lines.append(line)
                    continue
                x_center = (min_x + max_x) / 2.0
                y_center = (min_y + max_y) / 2.0
                w = max_x - min_x
                h = max_y - min_y
                out_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                changed = True
                continue
            print(f"{path}: line {i} has unexpected number of values ({len(vals)}), leaving as-is")
            out_lines.append(line)

        if changed:
            # bak = path.with_suffix(path.suffix + ".bak")
            # if not bak.exists():
            #     shutil.copy2(path, bak)
            path.write_text("\n".join(out_lines) + "\n")

    if labels_root.exists():
        for txt in labels_root.rglob("*.txt"):
            process_label_file(txt)

def down_LVIS(zipdir=None):
    coco_imgs = glob("datasets/COCO/images/train2017/*")
    if len(coco_imgs) < 100000:
        print("COCO dataset not found. Please download COCO dataset first.")
        return
    train_annotation_url = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
    val_annotation_url = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
    dir = Path("datasets/LVIS")
    if zipdir:
        zipdir.mkdir(parents=True, exist_ok=True)
        try:
            download([train_annotation_url, val_annotation_url], dir=zipdir / "annotations", curl=True, threads=2, exist_ok=True)
        except FileExistsError:
            download([train_annotation_url, val_annotation_url], dir=zipdir / "annotations", curl=True, threads=1, exist_ok=True)
        shutil.move(str(zipdir / "annotations"), str(dir / "annotations"))
    else:
        try:
            download([train_annotation_url, val_annotation_url], dir=dir / "annotations", curl=True, threads=2, exist_ok=True)
        except FileExistsError:
            download([train_annotation_url, val_annotation_url], dir=dir / "annotations", curl=True, threads=1, exist_ok=True)
    annotation_path = dir / "annotations"
    for zip_file in annotation_path.glob("*.zip"):
        zip_file.unlink()

    with open(dir / "annotations" / "lvis_v1_train.json") as f:
        train_data = json.load(f)
    with open(dir / "annotations" / "lvis_v1_val.json") as f:
        val_data = json.load(f)
    categories = train_data['categories']
    names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    cat_id_to_class = {cat['id']: i for i, cat in enumerate(sorted(categories, key=lambda x: x['id']))}
    
    yaml_path = dir / "data.yaml"
    data_yaml = []
    data_yaml.append(f"path: {dir}")
    data_yaml.append("train: images/train2017")
    data_yaml.append("val: images/val2017")
    data_yaml.append(f"nc: {len(names)}")
    data_yaml.append("names:")
    for n in names:
        data_yaml.append(f"  - {n}")
    yaml_path.write_text("\n".join(str(x) for x in data_yaml))
    print(f"data.yaml saved to: {yaml_path.resolve()}")
    
    def convert_bbox(img_w, img_h, bbox):
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        return x_center, y_center, w_norm, h_norm
    
    coco_train = Path("datasets/COCO/images/train2017").resolve()
    coco_val = Path("datasets/COCO/images/val2017").resolve()
    images_dir = dir / "images"
    labels_dir = dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    for split, data, coco_src in [('train', train_data, coco_train), ('val', val_data, coco_val)]:
        images_split = images_dir / f"{split}2017"
        labels_split = labels_dir / f"{split}2017"
        images_split.mkdir(exist_ok=True)
        labels_split.mkdir(exist_ok=True)
        
        ann_dict = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_dict:
                ann_dict[img_id] = []
            ann_dict[img_id].append(ann)
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Processing {split}", total=len(data['images']))
            for img in data['images']:
                img_id = img['id']
                file_name = img['coco_url'].rsplit('/', 1)[-1]
                src = coco_src / file_name
                if not src.exists():
                    if split == 'train':
                        src = coco_val / file_name
                    else:
                        src = coco_train / file_name
                dst_img = images_split / file_name
                os.symlink(src, dst_img)
                
                label_file = labels_split / file_name.replace('.jpg', '.txt')
                with open(label_file, 'w') as f:
                    if img_id in ann_dict:
                        for ann in ann_dict[img_id]:
                            cat_id = ann['category_id']
                            cls_id = cat_id_to_class[cat_id]
                            bbox = ann['bbox']
                            bb = convert_bbox(img['width'], img['height'], bbox)
                            f.write(f"{cls_id} {' '.join(map(str, bb))}\n")
                progress.update(task, advance=1)

def down_VisDrone(zipdir=None):
    def visdrone2yolo(dir, split, source_name=None):
        source_dir = dir / (source_name or f"VisDrone2019-DET-{split}")
        images_dir = dir / "images" / split
        labels_dir = dir / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)

        if (source_images_dir := source_dir / "images").exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            for img in source_images_dir.glob("*.jpg"):
                img.rename(images_dir / img.name)

        for f in tqdm((source_dir / "annotations").glob("*.txt"), desc=f"Converting {split}"):
            img_size = Image.open(images_dir / f.with_suffix(".jpg").name).size
            dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
            lines = []

            with open(f, encoding="utf-8") as file:
                for row in [x.split(",") for x in file.read().strip().splitlines()]:
                    if row[4] != "0":
                        x, y, w, h = map(int, row[:4])
                        cls = int(row[5]) - 1
                        x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                        w_norm, h_norm = w * dw, h * dh
                        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            (labels_dir / f.name).write_text("".join(lines), encoding="utf-8")


    dir = Path("datasets/VisDrone")
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
    ]
    if zipdir:
        zipdir.mkdir(parents=True, exist_ok=True)
        download(urls, dir=zipdir, threads=4)
        for folder in zipdir.glob("VisDrone*"):
            shutil.move(str(folder), str(dir / folder.name))
    else:
        download(urls, dir=dir, threads=4)

    splits = {"VisDrone2019-DET-train": "train", "VisDrone2019-DET-val": "val", "VisDrone2019-DET-test-dev": "test"}
    for folder, split in splits.items():
        visdrone2yolo(dir, split, folder)
        shutil.rmtree(dir / folder)

    yaml_path = dir / "data.yaml"
    data_yaml = []
    data_yaml.append(f"path: {dir}")
    data_yaml.append("train: images/train")
    data_yaml.append("val: images/val")
    data_yaml.append("test: images/test")
    data_yaml.append(f"nc: {len(VISDRONE_NAMES)}")
    data_yaml.append("names:")
    for n in VISDRONE_NAMES:
        data_yaml.append(f"  - {n}")
    yaml_path.write_text("\n".join(str(x) for x in data_yaml))
    print(f"data.yaml saved to: {yaml_path.resolve()}")
    for zip_file in dir.glob("*.zip"):
        zip_file.unlink()

def down_CUB200_2021(zipdir=None):
    dataset_kaggle_id = "dataset268/dataset-cub"
    download_dir = Path("datasets/CUB-200-2021")
    download_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    
    def save_sample(row, output_dir, yolo_class_id, cub_path):
        img_path = cub_path / 'images' / row['image_path']
        img_name = Path(row['image_path']).stem
        
        # Copy image to images directory
        shutil.copy2(img_path, output_dir / f"{img_name}.jpg")
        
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        x, y, w, h = row['x'], row['y'], row['width'], row['height']
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h
        
        # Write label to corresponding labels directory
        labels_dir = output_dir.parent.parent / "labels" / output_dir.name
        with open(labels_dir / f"{img_name}.txt", 'w') as f:
            f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def convert_cub_to_yolo(cub_path, output_path):
        cub = Path(cub_path)
        out = Path(output_path)
        
        (out / "images" / "train").mkdir(parents=True, exist_ok=True)
        (out / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Read available files
        images_df = pd.read_csv(cub / "images.txt", sep=' ', names=['image_id', 'image_path'], header=None)
        bbox_df = pd.read_csv(cub / "bounding_boxes.txt", sep=' ', names=['image_id', 'x', 'y', 'width', 'height'], header=None)
        
        # Extract class names from image paths
        images_df['class_name'] = images_df['image_path'].apply(lambda x: x.split('/')[0].split('.', 1)[1] if '.' in x.split('/')[0] else x.split('/')[0])
        unique_classes = sorted(images_df['class_name'].unique())
        class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
        
        # Create class_id column
        images_df['class_id'] = images_df['class_name'].map(class_to_id)
        
        # Merge with bounding boxes
        data = images_df.merge(bbox_df, on='image_id')
        
        # Create simple train/val split (80/20)
        data['is_training'] = data['image_id'] % 5 != 0  # Every 5th image for validation
        
        yaml_data = {
            'path': str(out),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(unique_classes),
            'names': unique_classes
        }
        
        with open(out / 'data.yaml', 'w') as f:
            yaml.dump(yaml_data, f)
        
        train_count = val_count = 0
        
        for _, row in data.iterrows():
            if row['is_training']:
                output_dir = out / "images" / "train"
                train_count += 1
            else:
                output_dir = out / "images" / "val"
                val_count += 1
            
            save_sample(row, output_dir, row['class_id'], cub)
        
        print(f"Converted {train_count} train, {val_count} val samples")
        return str(out)

    cub_extracted_path = download_dir / "CUB-200-2011"
    if cub_extracted_path.exists():
        print("CUB dataset already exists. Converting to YOLO format.")
    else:
        if zipdir:
            zipdir.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(dataset_kaggle_id, path=zipdir.as_posix(), unzip=False)
            zip_path = zipdir / "dataset-cub.zip"
        else:
            api.dataset_download_files(dataset_kaggle_id, path=download_dir.as_posix(), unzip=False)
            zip_path = download_dir / "dataset-cub.zip"
        
        download([str(zip_path)], dir=download_dir, curl=True, threads=1, exist_ok=True)
        
        if zip_path.exists():
            zip_path.unlink()
    
    convert_cub_to_yolo(cub_extracted_path, download_dir)
    shutil.rmtree(cub_extracted_path)

def down_kitti(zipdir=None):
    dataset_id = "taiducphan/kitti-yolo"
    download_dir = Path("datasets")
    download_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    if zipdir:
        zipdir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=zipdir.as_posix(), unzip=True)
        shutil.move(str(zipdir / "KITTI_YOLO"), str(download_dir / "KITTI_YOLO"))
    else:
        api.dataset_download_files(dataset_id, path=download_dir.as_posix(), unzip=True)
    os.system('mv datasets/KITTI_YOLO datasets/KITTI')
    yaml_data = {
        'path': 'datasets/KITTI',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 8,
        'names': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    }
    with open('datasets/KITTI/data.yaml', 'w') as f:
        yaml.dump(yaml_data, f)

def down_cityscapes(zipdir=None):
    dataset_id = "taiducphan/cityscapes-yolo"
    download_dir = Path("datasets")
    download_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    if zipdir:
        zipdir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=zipdir.as_posix(), unzip=True)
        shutil.move(str(zipdir / "Cityscapes_YOLO"), str(download_dir / "Cityscapes_YOLO"))
    else:
        api.dataset_download_files(dataset_id, path=download_dir.as_posix(), unzip=True)
    os.system('mv datasets/Cityscapes_YOLO datasets/Cityscapes')
    yaml_data = {
        'path': 'datasets/Cityscapes',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 28,
        'names': ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign', 'pole', 'building', 'wall', 'fence', 'vegetation', 'terrain', 'sky', 'road', 'sidewalk', 'parking', 'rail track', 'guard rail', 'bridge', 'tunnel', 'polegroup', 'caravan', 'trailer', 'license plate']
    }
    with open('datasets/Cityscapes/data.yaml', 'w') as f:
        yaml.dump(yaml_data, f)

def down_data(dataset_name, zipdir=None):
    if dataset_name == "VOC":
        down_VOC(zipdir)
    elif dataset_name.lower() == "coco":
        down_COCO(zipdir)
    elif dataset_name == "LVIS":
        down_LVIS(zipdir)
    elif dataset_name == "visdrone":
        down_VisDrone(zipdir)
    elif dataset_name == "cub":
        down_CUB200_2021(zipdir)
    elif dataset_name.upper() == "KITTI":
        down_kitti(zipdir)
    elif dataset_name.lower() == "cityscapes":
        down_cityscapes(zipdir)
    else:
        print(f"Dataset {dataset_name} not supported. Please choose from VOC, COCO, LVIS, visdrone, cub, KITTI, or Cityscapes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert datasets to YOLO format")
    parser.add_argument("--dataset", type=str, default="VOC", help="Name of the dataset (VOC, COCO, LVIS, visdrone, cub, KITTI, Cityscapes)")
    parser.add_argument("--zipdir", type=str, default=None, help="Directory to download zip files to before extracting to datasets")
    args = parser.parse_args()
    
    zipdir = Path(args.zipdir) if args.zipdir else None
    down_data(args.dataset, zipdir)