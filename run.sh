
OUTPUT_DIR="/nfs/home/theinerj/vid2pos_git-tib-other/cvsports22_challenge/sn-calib-challenge/data/segment_localization"

python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split valid --num_points_lines 2 --num_points_circles 8
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split test --num_points_lines 2 --num_points_circles 8
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split wc14-test --num_points_lines 2 --num_points_circles 8
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split valid --num_points_lines 4 --num_points_circles 8
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split test --num_points_lines 4 --num_points_circles 8
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split wc14-test --num_points_lines 4 --num_points_circles 8

# for DLT from line baseline -> best results for two major points

# ResNet-101 retrained
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split wc14-test --pp_num_points 2 --pp_maxdists 40
python custom_extremities.py -p $OUTPUT_DIR --checkpoint ../resources/train_59.pt --split test --pp_num_points 2 --pp_maxdists 40

# ResNet-50 original
python -m src.baseline_extremities --soccernet /nfs/data/soccernet/calibration/ --split wc14-test -p $OUTPUT_DIR/sn-baseline --masks true
python -m src.baseline_extremities --soccernet /nfs/data/soccernet/calibration/ --split test -p $OUTPUT_DIR/sn-baseline





