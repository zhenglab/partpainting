# Painting from Part

This repository provides the official PyTorch implementation of our paper "Painting from Part".

Our paper can be found in https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Painting_From_Part_ICCV_2021_paper.pdf.


## Prerequisites

- Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## Getting Started


### Installation

- Clone this repo:
```bash
git clone https://github.com/zhenglab/partpainting.git
cd partpainting
```

- Install [PyTorch](http://pytorch.org) and 1.7 and other dependencies (e.g., torchvision).
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.txt`.

### Training

Please change the pathes to your dataset path in `datasets` folder. We will update other used datasets train/test splitting files recently. 

The code defaults to regular outpainting task, and you may change mask types for other purpose in `src/dataset.py` and `src/utils.py`.

```
python train.py --path=$configpath$

For example: python train.py --path=./checkpoints/celeba-hq/
```

### Testing

The model is automatically saved every 50,000 iterations, please rename the file `g.pth_$iter_number$` to `g.pth` and then run testing command.
```
python test.py --path=$configpath$ 

For example: python test.py --path=./checkpoints/celeba-hq/
```

## Citing
```
@inproceedings{guo2020spiralnet,
author = {Guo, Dongsheng and Liu, Hongzhi and Zhao, Haoru and Cheng, Yunhao and Song, Qingwei and Gu, Zhaorui and Zheng, Haiyong and Zheng, Bing},
title = {Spiral Generative Network for Image Extrapolation},
booktitle = {The European Conference on Computer Vision (ECCV)},
pages={701--717},
year = {2020}
} 

```
