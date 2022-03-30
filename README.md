# iFS-RCNN: An Incremental Few-shot Instance Segmenter

This is the official code for the CVPR 2022 paper: ["iFS-RCNN: An Incremental Few-shot Instance Segmenter"](https://www.khoinguyen.org/publication/incremental-few-shot-instance-segmentation/khoi_cvpr2022_iFS_RCNN.pdf)

This codebase is primarily based on this codebase of https://github.com/ucbdrive/few-shot-object-detection


## Table of Contents
- [iFS-RCNN: An Incremental Few-shot Instance Segmenter](#ifs-rcnn-an-incremental-few-shot-instance-segmenter)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Code Structure](#code-structure)
  - [Data Preparation](#data-preparation)
  - [Getting Started](#getting-started)
    - [Training & Evaluation in Command Line](#training--evaluation-in-command-line)
    - [Multiple Runs](#multiple-runs)


## Installation

**Requirements**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 10.0, 10.1, 10.2
* GCC >= 4.9

**Build fsdet**
* Create a virtual environment.
```angular2html
python3 -m venv fsdet
source fsdet/bin/activate
```
You can also use `conda` to create a new environment.
```angular2html
conda create --name fsdet
conda activate fsdet
```
* Install Pytorch 1.6 with CUDA 10.2 
```angular2html
pip install torch torchvision
```
You can choose the Pytorch and CUDA version according to your machine.
Just to make sure your Pytorch version matches the [prebuilt detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#install-pre-built-detectron2-linux-only)
* Install Detectron2 v0.2.1
```angular2html
python3 -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
```
* Install other requirements. 
```angular2html
python3 -m pip install -r requirements.txt
```

## Code Structure
- **configs**: Configuration files
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **iFS-RCNN**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **engine**: Contains training and evaluation loops and hooks.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.


## Data Preparation
We evaluate our models on two datasets:
- [COCO](http://cocodataset.org/): We use COCO 2014 and extract 5k images from the val set for evaluation and use the rest for training. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.
- [LVIS](https://www.lvisdataset.org/): We treat the frequent and common classes as the base classes and the rare categories as the novel classes.

See [datasets/README.md](datasets/README.md) for more details.


## Getting Started


### Training & Evaluation in Command Line

+ To train a model on base classes, refer to the [train.sh](train.sh) script  

+ To fine-tune a model on novel classes, refer to the [finetuning.sh](finetuning.sh) script

+ To evaluate the trained models, refer to the [test.sh](test.sh) script 

### Multiple Runs

+ For multiple running experiments, refer to [run_experiments.sh](run_experiments.sh) script

+ To aggregate results of multiple runs, refer to [aggregate_seeds.sh](aggregate_seeds.sh) script
