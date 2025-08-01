<h1 align="center"> HoRoPose(Now with LeRobot SO100 support!): Real-time Holistic Robot Pose Estimation with Unknown States <br> (ECCV 2024)</h1>

<div align="center">

  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> 
  [![arXiv](https://img.shields.io/badge/arXiv-2402.05655-b31b1b.svg)](https://arxiv.org/abs/2402.05655.pdf) 
  <a href="https://oliverbansk.github.io/Holistic-Robot-Pose/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> 
  <a href="https://www.youtube.com/watch?v=9NsLJvp1IPE"><img alt="Video" src="https://img.shields.io/badge/-Video-ea3323?logo=youtube"></a>

  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/real-time-holistic-robot-pose-estimation-with/robot-pose-estimation-on-dream-dataset)](https://paperswithcode.com/sota/robot-pose-estimation-on-dream-dataset?p=real-time-holistic-robot-pose-estimation-with)

</div>

<img src="assets/holistic.gif" width="800"/>

This is the official PyTorch implementation of the paper "Real-time Holistic Robot Pose Estimation with Unknown States". It provides an efficient framework for <b>real-time</b> robot pose estimation from RGB images <b>without</b> requiring known robot states.
## Run Steps 1-3 on H100 GPU.
## 1. Creating HoRoPose Conda Environment

```bash
sudo apt update && sudo apt -y install git ffmpeg
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
bash mc.sh -b -p $HOME/miniconda && eval "$($HOME/miniconda/bin/conda shell.bash hook)"

conda create -y -n horopose python=3.10
conda activate horopose

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge opencv
```
## 2. Clone HoRoPose and Install Required Packages
```bash
git clone https://github.com/oliverbansk/holistic-robot-pose-estimation.git holorobot
cd holorobot
pip install -r requirements.txt        
pip install simplejson torchnet joblib yacs termcolor colorama transforms3d \
            pybullet xarray
```


## 3. Obtain so100 Urdf and tmp files
```bash
mkdir -p assets/so100
curl -L -o assets/so100/so100.urdf \   https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/master/Simulation/SO100/so100.urdf

git clone --depth 1 --filter=blob:none \
     https://github.com/TheRobotStudio/SO-ARM100 tmp
mv tmp/Simulation/SO100/assets/*.stl assets/so100/
rm -rf tmp
```
## Run steps 4-6 on Local Computer
## 4. Generate camera intrinsic and extrinsic jsons(Note convert all csv files into proper format(no spaces, quotes, brackets, or headings)
This step will be essential in creating the so100dreamdataset. This step utilizes a previously created checkerboard dataset and the checkerboard csv file. In simple and general terms, the checkerboard serves to capture all various possible positions of the so100 arm, which results in the prediction of the robot’s 3-D keypoints in the image frame.

To do this, first, you must run the calibrate_dual.py(so100_dream/calibrate_dual.py) script which uses checkered images(so100_dream/checkerboard_images) and their csv file(initial-episode-states.csv) to generate the intrinsics and extriniscs.

## 5. Generate the full so100dreamdataset used to train the HoRoPose motor position detection model
Run the make_dream.py script at so100_dream/make_dream.py(make sure you do this under Urdfpy_legacy conda environment: use python 3.8.20 because urdfpy only works with older python versions(like 3.8.20), then install urdfpy 0.0.22 package, everything else remains same as horopose conda environment) which uses images from 20 rgb videos and the 20rgb video csv file and urdf.

Run shell script commands to compile images(use convert_images_to_jpg.py) and compile json from make_dream.py into one dataset(rename files to keep it tidy).

Generates so100_dreamdataset_train for real images with ground truth.

## 6. Free Flowing steps(no specific order, just files need to be uploaded/changed)
Adapt depthnet and full.yaml files for the so100 arm, attach the so100_dreamdataset_train folder. For full.yaml, after training, make sure to put the correct path to the depthnet model. 

## Data and Model Preparation

In our work, we use the following data and pretrained model:
* The [DREAM datasets](https://drive.google.com/drive/folders/1uNK2n9wU4tRE07sM_r640wDhwmOwuxx6) consisting of both real and synthetic subsets, placed under `${ROOT}/data/dream/$`.
* The [URDF](https://drive.google.com/drive/folders/17KNhy28pypheYfDCxgOjJf4IyUnOI3gW?) (Unified Robotics Description Format) of robot Panda, Kuka and Baxter, placed under `${ROOT}/data/deps/$`.
* The [pretrained HRnet backbone](https://drive.google.com/file/d/1eqIftq1T_oIGhmCfkVYSM245Wj5xZaUo/view?) for pose estimation, placed under `${ROOT}/models/$`.
* The openly available [foreground segmentation model](https://drive.google.com/drive/folders/1PpXe3p5dJt9EOM-fwvJ9TNStTWTQFDNK?) of 4 real datasets of Panda from [CtRNet](https://github.com/ucsdarclab/CtRNet-robot-pose-estimation), placed under `${ROOT}/models/panda_segmentation/$`.

You can download the data and models through provided links. 
When finished, the directory tree should look like this. 
```
${ROOT}
|-- data
    |-- dream
    |   |-- real
    |   |   |-- panda-3cam_azure  
    |   |   |-- panda-3cam_kinect360
    |   |   |-- panda-3cam_realsense
    |   |   |-- panda-orb
    |   |-- synthetic
    |   |   |-- baxter_synth_test_dr
    |   |   |-- baxter_synth_train_dr
    |   |   |-- kuka_synth_test_dr
    |   |   |-- kuka_synth_test_photo
    |   |   |-- kuka_synth_train_dr
    |   |   |-- panda_synth_test_dr
    |   |   |-- panda_synth_test_photo
    |   |   |-- panda_synth_train_dr
    |-- deps
    |   |-- baxter-description
    |   |-- kuka-description
    |   |-- panda-description
|-- models
    |-- panda_segmentation
    |   |-- azure.pth
    |   |-- kinect.pth
    |   |-- orb.pth
    |   |-- realsense.pth
    |-- hrnet_w32-36af842e_roc.pth
```

## Train
We train our final model in a multi-stage fashion. All model is trained using a single NVIDIA V100 with 32GB GPU. Distributed training is also supported.

We use config files in `configs/` to specify the training process. We recommend filling in the `exp_name` field in the config files with a unique name, as the checkpoints and event logs produced during training will be saved under `experiments/{exp_name}`. The correspondent config file will be automatically copied into this directory.

### Synthetic Datasets

Firstly, pretrain the depthnet (root depth estimator) for 100 epochs for so100 arm:
```bash
python scripts/train.py --config configs/so100/depthnet.yaml

```

With depthnet pretrained, we can train the full network for 100 or 700 epochs:
```bash
python scripts/train.py --config configs/so100/full.yaml

```
To save your time when reproducing results of our paper, we provide readily-pretrained [depthnet model weights](https://drive.google.com/drive/folders/1rWC2bbA3U0IiZ7oDoKIVsWK_m4JkVarA?) for full network training. To use them, you can modify the `configs/{robot}/full.yaml` file by filling in the `pretrained_rootnet` field with the path of the downloaded `.pk` file. 

### Real Datasets of Panda

We employ self-supervised training for the 4 real datasets of Panda.

Firstly, train the model on synthetic dataset using `configs/panda/self_supervised/synth.yaml` for 100 epochs. Be sure to fill in the `pretrained_rootnet` field with the path of the pretrained Panda depthnet weight in advance.

```bash
python scripts/train.py --config configs/panda/self_supervised/synth.yaml
```
The training process above saves checkpoints for 4 real datasets for further self-supervised training (e.g. `experiments/{exp_name}/ckpt/curr_best_auc(add)_azure_model.pk`). 

When finished training on synthetic data, modify the `configs/panda/self_supervised/{real_dataset}.yaml` file by filling in the `pretrained_weight_on_synth` field with the path of the correspondent checkpoint. Then start self-supervised training with:

```bash
python scripts/train.py --config configs/panda/self_supervised/azure.yaml
python scripts/train.py --config configs/panda/self_supervised/kinect.yaml
python scripts/train.py --config configs/panda/self_supervised/realsense.yaml
python scripts/train.py --config configs/panda/self_supervised/orb.yaml
```

## Test
To evaluate models, simply run:
```bash
python scripts/test.py --exp_path {path of the experiment folder} --dataset {dataset name}
# e.g. python scripts/test.py -e experiments/panda_full --dataset panda_synth_test_dr
# You can add '--vis_skeleton' to visualize the robot keypoint skeleton
```
Note that each model is presented in a folder containing ckpt/, log/ and config.yaml. After running test script, result/ will be generated inside the folder.

## Model Zoo
You can download our final models from [Google Drive](https://drive.google.com/drive/folders/10Gz0NP39YyuvAlrhTa-XssWTDlyh9v80?usp=sharing) and evaluate them yourself.  


## Citation
If you use our code or models in your research, please cite with:
```bibtex
@inproceedings{holisticrobotpose,
  author={Ban, Shikun and Fan, Juling and Ma, Xiaoxuan and Zhu, Wentao and Qiao, Yu and Wang, Yizhou},
  title={Real-time Holistic Robot Pose Estimation with Unknown States},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}
```

## Acknowledgment
This repo is built on the excellent work [RoboPose](https://github.com/ylabbe/robopose) and [CtRNet](https://github.com/ucsdarclab/CtRNet-robot-pose-estimation). Thank the authors for releasing their codes.
