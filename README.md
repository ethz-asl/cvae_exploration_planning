# cvae_exploration_planning
**CVAE Exploration Planning** is a package providing an open-source implementation of the simulator, networks, and planners presented in our paper. TODO.


# Table of Contents
**Credits**
* [Paper and Video](#Paper-and-Video)

**Setup**
<!-- * [Packages](#Packages) -->
* [Dependencies](#Dependencies)
* [Installation](#Installation)
* [Data Repository](#Data-Repository)

**Examples**
* [Configuring a Planner](#Configuring-a-Planner)
* [Run an Experiment](#Run-an-Experiment)


# Credits
## Paper and Video
If you find this package useful for your research, please consider citing our paper:

* Lukas Schmid, Chao Ni, Yuliang Zhong, Roland Siegwart, and Olov Andersson, "**Fast and Compute-efficient Sampling-based Local Exploration Planning via Distribution Learning**", in *IEEE Robotics and Automation Letters*, vol. TODO, no. TODO, pp. TODO, June 2022 [IEEE | [ArXiv](https://arxiv.org/abs/2202.13715) | Video]
  ```bibtex
  @ARTICLE{Schmid22Fast,
    author={L. {Schmid} and C. {Ni} and Y. {Zhong} and and R. {Siegwart} and O. {Andersson}},
    journal={IEEE Robotics and Automation Letters},
    title={Fast and Compute-efficient Sampling-based Local Exploration Planning via Distribution Learning},
    year={2022},
    volume={?},
    number={?},
    pages={?},
    doi={?},
    ISSN={?},
    month={June},
  }
  ```
  
# Setup
We recommend using a virtual environment to run this project. We provide setup instructions using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) on Ubuntu.

**Note on versioning:** This repository was developed and tested using `Ubuntu 20.04` with `Python 3.8` and `Torch 1.7`. Other versions should also work.

## Dependencies

* Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and setup a virtual environment: 
  ```bash 
  conda create --name cvae 
  conda activate cvae
  ```

* Use conda to install [PyTorch](https://pytorch.org/) for your [cuda version](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html): 
  ```bash
  export MY_CUDA_VERSION='11.6' # Replace with your version. 
  conda install pytorch torchvision torchaudio cudatoolkit=$MY_CUDA_VERSION -c pytorch
  ```

## Installation

* Setup the destination where to isntall the project:
  ```bash
  export MY_CVAE_ROOT='/home/$USER/cvae_exploration_workspace' # Replace with your path.
  makedir -p $MY_CVAE_ROOT
  cd $MY_CVAE_ROOT
  ```

* Download the repository, we recommend using [SSH Keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#about-ssh-key-generation) or alternatively via HTTPS:
  ```bash
  git clone git@github.com:ethz-asl/cvae_exploration_planning.git # SSH
  git clone https://github.com/ethz-asl/cvae_exploration_planning.git # HTTPS
  cd cvae_exploration_planning
  ```

* Add the project folder to your python path:
  ```bash
  export PYTHONPATH=$MY_CVAE_ROOT
  ```

* You are now ready to go!

## Data-Repository

All our data is available on the [ETHZ ASL Dataserver](https://projects.asl.ethz.ch/datasets/doku.php?id=cvae_exploration_planning).
You can download all files needed to train and run our models and planners easily:
* Setup a data folder (this path is used by default):
  ```
  cd $MY_CVAE_ROOT/cvae_exploration_planning
  mkdir data
  ```
* Download the data used to train the CVAE models: 
  ```
  wget http://robotics.ethz.ch/~asl-datasets/2022_CVAE_Exploration_Planning/CVAE_dataset.npy -P data
  ```
* Download the data used to train the CNN models: 
  ```
  wget http://robotics.ethz.ch/~asl-datasets/2022_CVAE_Exploration_Planning/CNN_dataset.npy -P data
  ```
* Download the worlds used in our experiments: 
  ```
  wget http://robotics.ethz.ch/~asl-datasets/2022_CVAE_Exploration_Planning/test_worlds.zip
  unzip -j test_worlds.zip -d experiments/worlds
  rm test_worlds.zip
  ```

# Examples

To train the original CVAE model:

```
cd learning & python train.py
```

See `learning/config.yaml` for tunable parameters.

To train the two-stage model with a CNN based gain estimator:
```
cd learning & python train_cnn.py 
```
To evaluate the learned model, save the model to experiments/models, and

```
cd experiments & python evaluate.py
```
Choose the world, planner, runs in `experiments/config.yaml`.