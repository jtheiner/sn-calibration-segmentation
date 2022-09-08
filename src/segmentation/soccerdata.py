import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path
import json
import os

lines_classes = [
    'Big rect. left bottom',
    'Big rect. left main',
    'Big rect. left top',
    'Big rect. right bottom',
    'Big rect. right main',
    'Big rect. right top',
    'Circle central',
    'Circle left',
    'Circle right',
    'Goal left crossbar',
    'Goal left post left ',
    'Goal left post right',
    'Goal right crossbar',
    'Goal right post left',
    'Goal right post right',
    'Goal unknown',
    'Line unknown',
    'Middle line',
    'Side line bottom',
    'Side line left',
    'Side line right',
    'Side line top',
    'Small rect. left bottom',
    'Small rect. left main',
    'Small rect. left top',
    'Small rect. right bottom',
    'Small rect. right main',
    'Small rect. right top'
]

# RGB values
palette = {
    'Big rect. left bottom': (127, 0, 0),
    'Big rect. left main': (102, 102, 102),
    'Big rect. left top': (0, 0, 127),
    'Big rect. right bottom': (86, 32, 39),
    'Big rect. right main': (48, 77, 0),
    'Big rect. right top': (14, 97, 100),
    'Circle central': (0, 0, 255),
    'Circle left': (255, 127, 0),
    'Circle right': (0, 255, 255),
    'Goal left crossbar': (255, 255, 200),
    'Goal left post left ': (165, 255, 0),
    'Goal left post right': (155, 119, 45),
    'Goal right crossbar': (86, 32, 139),
    'Goal right post left': (196, 120, 153),
    'Goal right post right': (166, 36, 52),
    'Goal unknown': (0, 0, 0),
    'Line unknown': (0, 0, 0),
    'Middle line': (255, 255, 0),
    'Side line bottom': (255, 0, 255),
    'Side line left': (0, 255, 150),
    'Side line right': (0, 230, 0),
    'Side line top': (230, 0, 0),
    'Small rect. left bottom': (0, 150, 255),
    'Small rect. left main': (254, 173, 225),
    'Small rect. left top': (87, 72, 39),
    'Small rect. right bottom': (122, 0, 255),
    'Small rect. right main': (255, 255, 255),
    'Small rect. right top': (153, 23, 153)
}

data_dir = Path("data/datasets")

def create_target_from_annotation(width, height, annotation, classes):
    """Draw one-hot encoded segments according to the annotation.
    Creates target that matches image size ([C+1]xHxW).
    """
    annotation_abs = defaultdict(list)
    # unnormalize every point in every class k
    for k in annotation.keys():
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
            class_segments = cv2.line(class_segments, startxy, endxy, (1,1,1), 5)
        classes_segments[classes.index(cls) + 1] = class_segments[:,:,1]

    classes_segments = torch.Tensor(classes_segments)
    return classes_segments

class ExtremitiesDataset(Dataset):

    def __init__(self, root, split, classes=lines_classes, palette=palette):
        self.data_root = Path(root)
        self.split = split
        files = os.listdir(self.data_root / self.split)
        self.annotations = sorted([fn for fn in files if fn.endswith("json")])
        self.images = sorted([fn for fn in files if fn.endswith("jpg")])
        #self.height, self.width = 224, 224
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # see https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

        impath = self.data_root / self.split / self.images[idx]
        annotation_path = self.data_root / self.split / self.annotations[idx]
        #print(impath)
        #print(annotation_path)
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        
        # setup image, cast to device later in training
        img = Image.open(impath)
        trf = T.Compose(
            [
                T.Resize(256),
                #T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225]
                    )
            ]
        )
        # prepare batches
        img = trf(img)#.unsqueeze(0)
        new_height, new_width = img.shape[-2], img.shape[-1]
    
        
        target = create_target_from_annotation(new_width, new_height, annotation, self.classes)
        #target = torchvision.transforms.functional.center_crop(target, 224)
        target = target.long().argmax(dim=0)

        return img, target

if __name__ == "__main__":
    data = ExtremitiesDataset(root=data_dir, split="test")
    print(data[0][1])
    target = data[0][1].unsqueeze(0).permute(1,2,0)
    plt.imshow(target)
