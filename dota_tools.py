import os
from PIL import Image
import json
import numpy as np
import math
import pickle
from enum import IntEnum, unique

@unique
class BoxMode(IntEnum):
    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3
    XYWHA_ABS = 4

wordlist = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
]

def dota2detectron(dataset_path, use_cache=False, whitelist=wordlist):

    cache_path = os.path.join(dataset_path, "detectron_cache.p")
    if use_cache and os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))

    image_root = os.path.join(dataset_path, "images")
    label_root = os.path.join(dataset_path, "labelTxt")

    use_whitelist = True if len(whitelist) < len(wordlist) else False

    dataset_dicts = [] 
    for path in os.listdir(label_root):
        filename = os.path.basename(path)
        (p_id, ext) = os.path.splitext(filename)
        if ext != ".txt":
            continue

        image_filename = p_id + ".png"
        image_path= os.path.join(image_root, image_filename)
        img = Image.open(image_path)
        height, width = img.size

        image_dict = {
            "file_name": image_path,
            "image_id": p_id,
            "height": height,
            "width": width,
            "annotations": []
        }

        with open(os.path.join(label_root, filename), 'r') as fp:
            for line in fp: 
                splitlines = line.strip().split(" ")
                line_dict = {}
                # clear the wrong name after check all the data
                if len(splitlines) < 9:
                    continue
                if len(splitlines) >= 9:
                    line_dict["name"] = splitlines[8]
                if len(splitlines) == 9:
                    line_dict["difficult"] = "0"
                elif len(splitlines) >= 10:
                    line_dict["difficult"] = splitlines[9]
                # x1, y1, x2, y2, x3, y3, x4, y4,
                line_dict["poly"] = [float(splitlines[i]) for i in range(8)]
                if use_whitelist and line_dict["name"] not in whitelist:
                    continue

                # print(line_dict["name"])
                annotation = {
                    "category_id": whitelist.index(line_dict["name"]),
                    "bbox": polygon_to_xywha(line_dict["poly"]),
                    "bbox_mode": BoxMode.XYWHA_ABS,
                }
                image_dict["annotations"].append(annotation)


        n_annos = len(image_dict["annotations"])
        if n_annos > 3 and n_annos < 100:
            dataset_dicts.append(image_dict)

    # anno_len = [len(image_dict["annotations"]) for image_dict in dataset_dicts]
    # print(anno_len)

    pickle.dump(dataset_dicts, open(cache_path, "wb"))
    return dataset_dicts



def dota2coco(src_path, json_out_path):
    image_root = os.path.join(src_path, "images")
    label_root = os.path.join(src_path, "labelTxt")

    data_dict = {}
    info = {
        "description": "Imported DOTA dataset",
        "url": "http://captain.whu.edu.cn/DOTAweb/",
        "version": "1.5",
    }
    data_dict["info"] = info
    data_dict["images"] = []
    data_dict["categories"] = []
    data_dict["annotations"] = []

    for idex, name in enumerate(wordlist):
        single_cat = {"id": idex + 1, "name": name, "supercategory": name}
        data_dict["categories"].append(single_cat)

    inst_count = 1
    image_id = 1

    with open(json_out_path, "w") as f_out:
        # label_path_list = glob.glob(os.path.join(label_root, "*.txt")
        # label_path_list = ["test"]
        for path in os.listdir(label_root):
            filename = os.path.basename(path)
            (p_id, ext) = os.path.splitext(filename)
            if ext != ".txt":
                continue

            image_filename = p_id + ".png"
            image_path= os.path.join(image_root, image_filename)
            img = Image.open(image_path)
            height, width = img.size

            image_dict = {}
            image_dict["file_name"] = image_filename
            image_dict["id"] = image_id
            image_dict["width"] = width
            image_dict["height"] = height
            data_dict["images"].append(image_dict)

            # annotations
            with open(os.path.join(label_root, filename), 'r') as fp:
                for line in fp: 
                    splitlines = line.strip().split(" ")
                    line_dict = {}
                    # clear the wrong name after check all the data
                    if len(splitlines) < 9:
                        continue
                    if len(splitlines) >= 9:
                        line_dict["name"] = splitlines[8]
                    if len(splitlines) == 9:
                        line_dict["difficult"] = "0"
                    elif len(splitlines) >= 10:
                        line_dict["difficult"] = splitlines[9]
                    # x1, y1, x2, y2, x3, y3, x4, y4,
                    line_dict["poly"] = [
                        float(splitlines[0]),
                        float(splitlines[1]),
                        float(splitlines[2]),
                        float(splitlines[3]),
                        float(splitlines[4]),
                        float(splitlines[5]),
                        float(splitlines[6]),
                        float(splitlines[7]),
                    ]

                    annotation = {
                        "category_id": wordlist.index(line_dict["name"]) + 1,
                        "bbox": polygon_to_xywha(line_dict["poly"]),
                        "bbox_mode": 4,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "id": inst_count
                    }
                    # annotation["area"] = line_dict["area"]
                    # annotation["segmentation"] = []
                    # annotation["segmentation"].append(line_dict["poly"])

                    data_dict["annotations"].append(annotation)
                    inst_count += 1
            image_id += 1
        json.dump(data_dict, f_out)

def ang_diff(a, b):
    d = a - b
    return (d + 90) % 180 - 90 


def polygon_to_xywha(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    # reshpae to [[x0,...],[y0,...]]
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.reshape(bbox, newshape=(2, 4), order="F")

    bbox_shift = np.roll(bbox, 1, axis=1) 
    dx = - (bbox[0,:] - bbox_shift[0,:])
    dy = bbox[1,:] - bbox_shift[1,:]
    sides = np.hypot(dx, dy) # side lengths

    angles = np.arctan2(dy, dx)
    # angles[[1,3]] = np.pi / 2.0 - angles[[1,3]] # FIXME should also be done wirth atan2?
    angle = angles[[0,2]].mean()


    center = np.rint(np.mean(bbox, axis=1))
    wh = np.rint((sides[0:2] + sides[2:4]) / 2.0) # round mean to nearest int
    # print (np.rad2deg(angle))

    return [center[0],center[1], wh[1], wh[0], np.rad2deg(angle)]

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from rotated_detectron_utils import RotatedVisualizer
    from detectron2.data import DatasetCatalog, MetadataCatalog

    # TRAIN_PATH = "/data2/anne/UIVER/datasets/DOTA/dotadevkit/example"
    TRAIN_PATH = "/data2/anne/UIVER/datasets/DOTA/trainsplit"
    outfile = "datasets/DOTA/train/coco_test.json"

    
    class_name_list = ["small-vehicle", "large-vehicle"]

    DatasetCatalog.clear()
    MetadataCatalog.clear()
    DatasetCatalog.register("Train", lambda: dota2detectron(TRAIN_PATH, False, whitelist=class_name_list))
    MetadataCatalog.get("Train").set(thing_classes=class_name_list)

    train_catalog = dota2detectron(TRAIN_PATH, False, whitelist=class_name_list)
    d = train_catalog[2]

    img = cv2.imread(d["file_name"])
    visualizer = RotatedVisualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("Train"), scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.savefig("dummy_image.png", dpi=350)
    # print(d['annotations'])
    for dat in train_catalog:
        for anno in dat['annotations']:
            pass
            # print(anno['bbox'][4])


    # bbox = [1054.0, 1028.0, 1063.0, 1011.0, 1111.0, 1040.0, 1112.0, 1062.0]
    bbox = [2102 ,1823 ,2124 ,1840 ,1862, 2233, 1839, 2217]
    polygon_to_xywha(bbox)
    

