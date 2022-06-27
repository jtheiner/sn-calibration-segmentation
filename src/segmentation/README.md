# Semantic segmentation reference training scripts

This directory is an edited copy of the [torchvision](https://github.com/pytorch/vision/tree/main/references/segmentation) reference scripts.

# Usage

Train from scratch:

```
python train.py -b 8 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --epochs 30 --output-dir "./checkpoints" --split train
```

Resume training:

```
python train.py -b 8 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --epochs 60 --start-epoch 30 --weights /path/to/checkpoints.pt
```

Evaluate checkpoint:

```
python train.py -b 4 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --test-only --weights /path/to/checkpoints.pt
```