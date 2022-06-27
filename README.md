# EVS Camera Calibration Challenge

This is a fork of the [EVS Camera Calibration Challenge repository](https://github.com/SoccerNet/sn-calibration).

## Usage

Train a new model using the src/segmentation directory.

Extract extremities keypoints using `custom_extremities.py`.

Overview of useful arguments:

 * `-p`: Output directory name for predictions (and masks).
    * Note: The final path for outputs will be extended: `DIRNAME_np{args.pp_num_points}_r{args.pp_radius}_md{args.pp_maxdists} / args.split`
 * `--split`: Select data split from base directory
 * `--checkpoint`: Model checkpoint for self-trained network.
 * `--pp_radius`: Post processing: Radius of circles that cover each segment. (Default changed from 6 to 4)
 * `--pp_maxdists`: Post processing: Maximum distance of circles that are allowed within one segment. (Default changed from 40 to 30)
 * `--pp_num_points`: Post processing: Number of keypoints that represent a segment.

Example:

```bash
sn-calibration/src> python custom_extremities.py -p pitchloc_train --checkpoint ../resources/train_59.pt --split test --pp_num_points 4

```

