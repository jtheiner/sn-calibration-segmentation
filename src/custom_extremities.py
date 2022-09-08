import argparse
import copy
import itertools
import json
import os.path
import random
from collections import deque
from pathlib import Path

from pytorch_lightning import seed_everything

seed_everything(seed=10, workers=True)

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm

from SoccerNet.Evaluation.utils_calibration import SoccerPitch


def generate_class_synthesis(semantic_mask, radius):
    """
    This function selects for each class present in the semantic mask, a set of circles that cover most of the semantic
    class blobs.
    :param semantic_mask: a image containing the segmentation predictions
    :param radius: circle radius
    :return: a dictionary which associates with each class detected a list of points ( the circles centers)
    """
    buckets = dict()
    kernel = np.ones((5, 5), np.uint8)
    semantic_mask = cv.erode(semantic_mask, kernel, iterations=1)
    for k, class_name in enumerate(SoccerPitch.lines_classes):
        mask = semantic_mask == k + 1
        if mask.sum() > 0:
            disk_list = synthesize_mask(mask, radius)
            if len(disk_list):
                buckets[class_name] = disk_list

    return buckets


def join_points(point_list, maxdist):
    """
    Given a list of points that were extracted from the blobs belonging to a same semantic class, this function creates
    polylines by linking close points together if their distance is below the maxdist threshold.
    :param point_list: List of points of the same line class
    :param maxdist: minimal distance between two polylines.
    :return: a list of polylines
    """
    polylines = []

    if not len(point_list):
        return polylines
    head = point_list[0]
    tail = point_list[0]
    polyline = deque()
    polyline.append(point_list[0])
    remaining_points = copy.deepcopy(point_list[1:])

    while len(remaining_points) > 0:
        min_dist_tail = 1000
        min_dist_head = 1000
        best_head = -1
        best_tail = -1
        for j, point in enumerate(remaining_points):
            dist_tail = np.sqrt(np.sum(np.square(point - tail)))
            dist_head = np.sqrt(np.sum(np.square(point - head)))
            if dist_tail < min_dist_tail:
                min_dist_tail = dist_tail
                best_tail = j
            if dist_head < min_dist_head:
                min_dist_head = dist_head
                best_head = j

        if min_dist_head <= min_dist_tail and min_dist_head < maxdist:
            polyline.appendleft(remaining_points[best_head])
            head = polyline[0]
            remaining_points.pop(best_head)
        elif min_dist_tail < min_dist_head and min_dist_tail < maxdist:
            polyline.append(remaining_points[best_tail])
            tail = polyline[-1]
            remaining_points.pop(best_tail)
        else:
            polylines.append(list(polyline.copy()))
            head = remaining_points[0]
            tail = remaining_points[0]
            polyline = deque()
            polyline.append(head)
            remaining_points.pop(0)
    polylines.append(list(polyline))
    return polylines


def get_line_extremities(buckets, maxdist, width, height, num_points_lines, num_points_circles):
    """
    Given the dictionary {lines_class: points}, finds plausible extremities of each line, i.e the extremities
    of the longest polyline that can be built on the class blobs,  and normalize its coordinates
    by the image size.
    :param buckets: The dictionary associating line classes to the set of circle centers that covers best the class
    prediction blobs in the segmentation mask
    :param maxdist: the maximal distance between two circle centers belonging to the same blob (heuristic)
    :param width: image width
    :param height: image height
    :return: a dictionary associating to each class its extremities
    """
    extremities = dict()
    for class_name, disks_list in buckets.items():
        polyline_list = join_points(disks_list, maxdist)
        max_len = 0
        longest_polyline = []
        for polyline in polyline_list:
            if len(polyline) > max_len:
                max_len = len(polyline)
                longest_polyline = polyline
        extremities[class_name] = [
            {'x': longest_polyline[0][1] / width, 'y': longest_polyline[0][0] / height},
            {'x': longest_polyline[-1][1] / width, 'y': longest_polyline[-1][0] / height}, 
            
        ]
        num_points = num_points_lines
        if "Circle" in class_name:
            num_points = num_points_circles
        if num_points > 2:
            # equally spaced points along the longest polyline
            # skip first and last as they already exist
            for i in range(1, num_points - 1):
                extremities[class_name].insert(
                    len(extremities[class_name]) - 1,
                    {'x': longest_polyline[i * int(len(longest_polyline) / num_points)][1] / width, 'y': longest_polyline[i * int(len(longest_polyline) / num_points)][0] / height}
                )

    return extremities


def get_support_center(mask, start, disk_radius, min_support=0.1):
    """
    Returns the barycenter of the True pixels under the area of the mask delimited by the circle of center start and
    radius of disk_radius pixels.
    :param mask: Boolean mask
    :param start: A point located on a true pixel of the mask
    :param disk_radius: the radius of the circles
    :param min_support: proportion of the area under the circle area that should be True in order to get enough support
    :return: A boolean indicating if there is enough support in the circle area, the barycenter of the True pixels under
     the circle
    """
    x = int(start[0])
    y = int(start[1])
    support_pixels = 1
    result = [x, y]
    xstart = x - disk_radius
    if xstart < 0:
        xstart = 0
    xend = x + disk_radius
    if xend > mask.shape[0]:
        xend = mask.shape[0] - 1

    ystart = y - disk_radius
    if ystart < 0:
        ystart = 0
    yend = y + disk_radius
    if yend > mask.shape[1]:
        yend = mask.shape[1] - 1

    for i in range(xstart, xend + 1):
        for j in range(ystart, yend + 1):
            dist = np.sqrt(np.square(x - i) + np.square(y - j))
            if dist < disk_radius and mask[i, j] > 0:
                support_pixels += 1
                result[0] += i
                result[1] += j
    support = True
    if support_pixels < min_support * np.square(disk_radius) * np.pi:
        support = False

    result = np.array(result)
    result = np.true_divide(result, support_pixels)

    return support, result


def synthesize_mask(semantic_mask, disk_radius):
    """
    Fits circles on the True pixels of the mask and returns those which have enough support : meaning that the
    proportion of the area of the circle covering True pixels is higher that a certain threshold in order to avoid
    fitting circles on alone pixels.
    :param semantic_mask: boolean mask
    :param disk_radius: radius of the circles
    :return: a list of disk centers, that have enough support
    """
    mask = semantic_mask.copy().astype(np.uint8)
    points = np.transpose(np.nonzero(mask))
    disks = []
    while len(points):

        start = random.choice(points)
        dist = 10.
        success = True
        while dist > 1.:
            enough_support, center = get_support_center(mask, start, disk_radius)
            if not enough_support:
                bad_point = np.round(center).astype(np.int32)
                cv.circle(mask, (bad_point[1], bad_point[0]), disk_radius, (0), -1)
                success = False
            dist = np.sqrt(np.sum(np.square(center - start)))
            start = center
        if success:
            disks.append(np.round(start).astype(np.int32))
            cv.circle(mask, (disks[-1][1], disks[-1][0]), disk_radius, 0, -1)
        points = np.transpose(np.nonzero(mask))

    return disks

class CustomNetwork:

    def __init__(self, checkpoint):
        print("Loading model" + checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = deeplabv3_resnet101(num_classes=len(SoccerPitch.lines_classes) + 1, aux_loss=True)
        self.model.load_state_dict(torch.load(checkpoint)["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("using", self.device)

    def forward(self, img):
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
        img = trf(img).unsqueeze(0).to(self.device) 
        result = self.model(img)["out"].detach().squeeze(0).argmax(0)
        result = result.cpu().numpy().astype(np.uint8)
        #print(result)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('-s', '--soccernet', default="/nfs/data/soccernet/calibration/", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('-p', '--prediction', default="sn-calib-test_endpoints", required=False, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('--split', required=False, type=str, default="challenge", help='Select the split of data')
    parser.add_argument('--masks', required=False, type=bool, default=False, help='Save masks in prediction directory')
    parser.add_argument('--resolution_width', required=False, type=int, default=455,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=256,
                        help='height resolution of the images')
    parser.add_argument('--checkpoint', required=False, type=str, help="Path to the custom model checkpoint.")
    parser.add_argument('--pp_radius', required=False, type=int, default=4,
                        help='Post processing: Radius of circles that cover each segment.')
    parser.add_argument('--pp_maxdists', required=False, type=int, default=30,
                        help='Post processing: Maximum distance of circles that are allowed within one segment.')
    parser.add_argument('--num_points_lines', required=False, type=int, default=2, choices=range(2,10),
                        help='Post processing: Number of keypoints that represent a line segment')
    parser.add_argument('--num_points_circles', required=False, type=int, default=2, choices=range(2,10),
                        help='Post processing: Number of keypoints that represent a circle segment')
    args = parser.parse_args()

    lines_palette = [0, 0, 0]
    for line_class in SoccerPitch.lines_classes:
        lines_palette.extend(SoccerPitch.palette[line_class])

    model = CustomNetwork(args.checkpoint)

    dataset_dir = os.path.join(args.soccernet, args.split)
    if not os.path.exists(dataset_dir):
        print("Invalid dataset path !")
        exit(-1)

    radius = args.pp_radius
    maxdists = args.pp_maxdists
    
    frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]
    with tqdm(enumerate(frames), total=len(frames), ncols=160) as t:
        for i, frame in t:

            output_prediction_folder = os.path.join(str(args.prediction), f"np{args.num_points_lines}_nc{args.num_points_circles}_r{radius}_md{maxdists}", args.split)
            if not os.path.exists(output_prediction_folder):
                os.makedirs(output_prediction_folder)
            prediction = dict()
            count = 0

            frame_path = os.path.join(dataset_dir, frame)

            frame_index = frame.split(".")[0]

            image = Image.open(frame_path)

            semlines = model.forward(image)
            #print(semlines.shape)
            # print("\nsemlines", type(semlines), semlines.shape)
            if args.masks:
                mask = Image.fromarray(semlines.astype(np.uint8)).convert('P')
                mask.putpalette(lines_palette)
                mask_file = os.path.join(output_prediction_folder, frame)
                mask.convert("RGB").save(mask_file)
            skeletons = generate_class_synthesis(semlines, radius)

            extremities = get_line_extremities(skeletons, maxdists, args.resolution_width, args.resolution_height, args.num_points_lines, args.num_points_circles)


            prediction = extremities
            count += 1

            prediction_file = os.path.join(output_prediction_folder, f"extremities_{frame_index}.json")
            with open(prediction_file, "w") as f:
                json.dump(prediction, f, indent=4)

