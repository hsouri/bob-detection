# BoB-Detection

This repository is the official implementation of <strong>Object Detection and Instance Segmentation</strong> task in the [*Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks*](https://github.com/hsouri/Battle-of-the-Backbones).

:pushpin: Our implementation and instructions are based on [mmdetection](https://github.com/open-mmlab/mmdetection)

## Installation

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

**Step 3.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full==1.7.0
```

**Step 4.** Install BoB-Detection.

```shell
git clone https://github.com/hsouri/bob-detection.git
cd bob-detection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 5.** Download COCO (LVIS) and unzip dataset (you can optionally delete downloaded zip files by passing '--delete').

COCO download:

```shell
python tools/misc/download_dataset.py --dataset-name coco2017 --unzip
```
LVIS download:

```shell
python tools/misc/download_dataset.py --dataset-name lvis --save-dir data/lvis_v1/ --unzip
cd data/lvis_v1/
mkdir annotations
mv lvis_v1_train.json annotations/
mv lvis_v1_val.json annotations/
```

Please refer to [Get Started](https://github.com/hsouri/bob-detection/blob/master/docs/en/get_started.md), [Dataset Prepare](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html?highlight=dataset), and [Dataset Download](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download) for more detailed instructions.

## Usage

The config files for all experiments in <strong>Battle of the Backbones (BoB)</strong> can be found [configs/bob](https://github.com/hsouri/bob-detection/tree/main/configs/bob).

To train a detector with the existing configs, run:

```shell
bash ./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
```
