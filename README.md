# SimMatch: Semi-supervised Learning with Similarity Matching

This repository contains PyTorch evaluation code, training code and pretrained models for SimMatch. Most of the code in this repository is adapt from [here](https://github.com/amazon-research/exponential-moving-average-normalization).

## Reproducing
To run the code, you probably need to change the Dataset setting (ImagenetPercentV2 function in dataset/imagenet.py), and Pytorch DDP setting (dist_init function in util/dist_utils.py) for your server environment.

The distributed training of this code is based on slurm environment, we have provided the training scrips int script/train.sh

We also provide the pre-trained model. 

|          |Arch | Setting | Epochs  | Accuracy | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|
|  ReSSL | ResNet50 | 1% | 400  | 67.2 % | [simmatch-1p.pth]() |
|  ReSSL | ResNet50 | 10% | 400  | 74.4 % | [simmatch-10p.pth]() |

If you want to test the pre-trained model, please download the weights from the link above, and move them to the checkpoints folder. The evaluation scripts also have been provided in script/train.sh
