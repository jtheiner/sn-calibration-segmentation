import argparse
import os.path
import pickle

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

from SoccerNet.Evaluation.utils_calibration import SoccerPitch

from custom_extremities import CustomNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument(
        "-s",
        "--soccernet",
        default="/nfs/data/soccernet/calibration/",
        type=str,
        help="Path to the SoccerNet-V3 dataset folder",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        default="/nfs/home/rhotertj/datasets/sn-calib-test_endpoints",
        required=False,
        type=str,
        help="Path to the prediction folder",
    )
    parser.add_argument(
        "--split",
        required=False,
        type=str,
        default="challenge",
        help="Select the split of data",
    )
    parser.add_argument(
        "--resolution_width",
        required=False,
        type=int,
        default=455,
        help="width resolution of the images",
    )
    parser.add_argument(
        "--resolution_height",
        required=False,
        type=int,
        default=256,
        help="height resolution of the images",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        type=str,
        help="Path to the custom model checkpoint.",
    )
    parser.add_argument("--filter_cam", type=str, required=False)
    args = parser.parse_args()

    lines_palette = [0, 0, 0]
    for line_class in SoccerPitch.lines_classes:
        print(line_class, SoccerPitch.palette[line_class])
        lines_palette.extend(SoccerPitch.palette[line_class])

    print(lines_palette)

    # exit(0)

    dataset_dir = os.path.join(args.soccernet, args.split)
    if not os.path.exists(dataset_dir):
        print("Invalid dataset path !")
        exit(-1)

    match_info_file = os.path.join(args.soccernet, args.split, "match_info_cam_gt.json")
    print(match_info_file)
    if not os.path.exists(match_info_file):
        exit(-1)
    df = pd.read_json(match_info_file).T
    if args.filter_cam:
        df = df.loc[df.camera == args.filter_cam]
    df["image_file"] = df.index
    df = df.sort_values(by=["image_file"])
    print(df)

    frames = df["image_file"].tolist()

    model = CustomNetwork(args.checkpoint)

    image_src = []
    edge_maps = np.zeros((len(frames), 1, 180, 320), dtype=np.uint8)

    kernel = np.ones((4, 4), np.uint8)

    with tqdm(enumerate(frames), total=len(frames), ncols=100) as t:
        for i, frame in t:

            output_prediction_folder = args.prediction
            if not os.path.exists(output_prediction_folder):
                os.makedirs(output_prediction_folder)

            frame_path = os.path.join(dataset_dir, frame)

            frame_index = frame.split(".")[0]

            image = Image.open(frame_path)

            semlines = model.forward(image)

            # print(semlines.shape, np.unique(semlines))
            # set class 9-15 (goal parts) to background
            mask_goal = (semlines >= 9) & (semlines <= 15)
            semlines[mask_goal] = 0

            mask = Image.fromarray(semlines.astype(np.uint8)).convert("P")
            mask.putpalette(lines_palette)

            # to binary edge map
            mask = np.asarray(mask.convert("L"))
            mask[mask > 0] = 255

            mask = Image.fromarray(mask)
            mask = mask.resize((320, 180), resample=Image.NEAREST)
            # expected linewith @ 720p resulution -> 4px

            mask = np.asarray(mask)
            # print(mask.shape)

            mask = cv2.erode(mask, kernel, iterations=1)

            # assert len(np.unique(mask)) == 2  # [0, 255]

            # mask_file = os.path.join(output_prediction_folder, frame)
            # mask.save(mask_file)
            # print(mask)
            # exit(0)

            edge_maps[i] = mask
            image_src.append(frame)

    with h5py.File(
        os.path.join(output_prediction_folder, "seg_edge_maps.h5"), "w"
    ) as f:
        f.create_dataset("edge_map", data=edge_maps)

    with open(os.path.join(output_prediction_folder, "seg_image_paths.pkl"), "wb") as f:
        pickle.dump(image_src, f)
