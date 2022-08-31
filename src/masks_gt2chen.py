import pickle
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from tqdm.auto import tqdm
from collections import defaultdict
from pathlib import Path
import json
import os
from argparse import ArgumentParser

lines_classes = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    # "Goal left crossbar",
    # "Goal left post left ",
    # "Goal left post right",
    # "Goal right crossbar",
    # "Goal right post left",
    # "Goal right post right",
    "Goal unknown",
    "Line unknown",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
]

# RGB values
palette = {
    "Big rect. left bottom": (127, 0, 0),
    "Big rect. left main": (102, 102, 102),
    "Big rect. left top": (0, 0, 127),
    "Big rect. right bottom": (86, 32, 39),
    "Big rect. right main": (48, 77, 0),
    "Big rect. right top": (14, 97, 100),
    "Circle central": (0, 0, 255),
    "Circle left": (255, 127, 0),
    "Circle right": (0, 255, 255),
    # "Goal left crossbar": (255, 255, 200),
    # "Goal left post left ": (165, 255, 0),
    # "Goal left post right": (155, 119, 45),
    # "Goal right crossbar": (86, 32, 139),
    # "Goal right post left": (196, 120, 153),
    # "Goal right post right": (166, 36, 52),
    "Goal unknown": (0, 0, 0),
    "Line unknown": (0, 0, 0),
    "Middle line": (255, 255, 0),
    "Side line bottom": (255, 0, 255),
    "Side line left": (0, 255, 150),
    "Side line right": (0, 230, 0),
    "Side line top": (230, 0, 0),
    "Small rect. left bottom": (0, 150, 255),
    "Small rect. left main": (254, 173, 225),
    "Small rect. left top": (87, 72, 39),
    "Small rect. right bottom": (122, 0, 255),
    "Small rect. right main": (255, 255, 255),
    "Small rect. right top": (153, 23, 153),
}


def create_target_from_annotation(width, height, annotation, classes, linewidth=4):
    """Draw one-hot encoded segments according to the annotation.
    Creates target that matches image size ([C+1]xHxW).
    """
    annotation_abs = defaultdict(list)
    # unnormalize every point in every class k
    for k in annotation.keys():
        if k not in lines_classes:
            continue
        start = annotation[k][0].copy()
        end = annotation[k][-1].copy()
        for annotation_point in annotation[k]:
            tup = annotation_point.copy()
            tup["x"] *= width
            tup["x"] = int(tup["x"])
            tup["y"] *= height
            tup["y"] = int(tup["y"])
            annotation_abs[k].append(tup)

    # draw lines between annotated points for each segment
    # offset class +1 such that no classes detected will end in argmax 0
    # otherwise argmax 0 will be another class
    classes_segments = np.zeros(shape=(len(classes) + 1, height, width))
    for cls, points in annotation_abs.items():
        class_segments = np.zeros(shape=(height, width, 3))
        for start, end in zip(points, points[1:]):
            startxy = (start["x"], start["y"])
            endxy = [end["x"], end["y"]]
            class_segments = cv2.line(
                class_segments, startxy, endxy, (1, 1, 1), linewidth
            )
        classes_segments[classes.index(cls) + 1] = class_segments[:, :, 1]

    classes_segments = torch.Tensor(classes_segments)
    return classes_segments


class ExtremitiesDataset(Dataset):
    def __init__(
        self, root, split, annotations, filter_cam=None, extremities_prefix="", classes=lines_classes, palette=palette
    ):
        self.data_root = Path(root)
        self.split = split

        self.annotations_path = annotations

        if filter_cam is None:
            files = os.listdir(self.data_root / self.split)
            self.annotations = sorted([fn for fn in files if fn.endswith("json")])
            self.images = sorted([fn for fn in files if fn.endswith("jpg")])
        else:
            df = pd.read_json(self.data_root / self.split / "match_info_cam_gt.json").T
            df = df.loc[df.camera == filter_cam]
            assert len(df.index) > 0
            df["image_file"] = df.index
            df = df.sort_values(by=["image_file"])
            df["annotation_file"] = df["image_file"].apply(
                lambda s: extremities_prefix + s.split(".jpg")[0] + ".json"
            )
            self.annotations = df["annotation_file"].tolist()
            self.images = df["image_file"].tolist()

        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # see https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

        impath = self.data_root / self.split / self.images[idx]
        annotation_path = self.annotations_path /  self.annotations[idx]
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        img = Image.open(impath)  # .resize((1280, 720))
        trf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # prepare batches
        img = trf(img)

        # see https://git.tib.eu/vid2pos/sccvsd/-/blob/master/utils/synthetic_util.py
        # draw lines (linewidth=4 for 720p) -> hence we rescale first
        target = create_target_from_annotation(1280, 720, annotation, self.classes)
        target = target.long().argmax(dim=0).unsqueeze(0)
        # to binary mask
        target = target.bool().float()
        # rescale target equivalent to cv2.resize() with default args (interpolation bilinear) -> same as in torchvision
        # bilinear -> [0, 0.25, 0.5, 0.7, 1.0] are entries
        target = torchvision.transforms.Resize((180, 320))(target)
        # print(torch.unique(target))
        # to uint8
        target = (target * 255.0).to(torch.uint8)

        return img, target, impath.name


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--data_dir", type=Path)
    args.add_argument("--annotations", type=Path)
    args.add_argument("--output_dir", type=Path)
    args.add_argument("--extremities_prefix", type=str, default="")
    args = args.parse_args()

    data_dir = args.data_dir.parent
    split = args.data_dir.name
    output_dir = args.output_dir
    if not output_dir.exists():
        raise FileNotFoundError

    dataset = ExtremitiesDataset(data_dir, split, args.annotations, filter_cam="Main camera center", extremities_prefix=args.extremities_prefix)

    # img, edge_map, img_id = dataset[0]
    # edge_map = edge_map.squeeze(0).numpy()
    # Image.fromarray(edge_map).show()

    image_src = []
    edge_maps = np.zeros((len(dataset), 1, 180, 320), dtype=np.uint8)
    for i, (_, edge_map, img_id) in enumerate(tqdm(dataset)):
        edge_map = edge_map.numpy()
        # Image.fromarray(edge_map).show()
        edge_maps[i] = edge_map
        image_src.append(img_id)

    with h5py.File(output_dir / "seg_edge_maps.h5", "w") as f:
        f.create_dataset("edge_map", data=edge_maps)

    with open(output_dir / "seg_image_paths.pkl", "wb") as f:
        pickle.dump(image_src, f)
