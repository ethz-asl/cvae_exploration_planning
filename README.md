# cvae_exploration_planning
**CVAE Exploration Planning** is proposes a new approach to local exploration planning by combining learning and the sampling-based planning paradigm! This package provides an open-source implementation of the simulator, datasets, models, and planners presented in our paper on learning sampling-based local exploration.

![CVAE Exploration Planning](https://user-images.githubusercontent.com/36043993/176672800-24e26425-ae9b-4df9-98f3-70b71d3cd2c7.gif)

# Table of Contents
**Credits**
* [Paper and Video](#Paper-and-Video)

**Setup**
* [Dependencies](#Dependencies)
* [Installation](#Installation)
* [Data Repository](#Data-Repository)

**Examples**
* [Training the CVAE Model](#Training-the-CVAE-Model)
* [Training the CNN Model](#Training-the-CNN-Model)
* [Evaluating a Planner](#Evaluating-a-Planner)
* [Using the simulator](#Using-the-simulator)

**Additional Information**
* [Project Overview](#Project-Overview)


# Credits
## Paper and Video
If you find this package useful for your research, please consider citing our paper:

* Lukas Schmid, Chao Ni, Yuliang Zhong, Roland Siegwart, and Olov Andersson, "**Fast and Compute-efficient Sampling-based Local Exploration Planning via Distribution Learning**", in *IEEE Robotics and Automation Letters*, June 2022 [IEEE | [ArXiv](https://arxiv.org/abs/2202.13715) | Video]
  ```bibtex
  @ARTICLE{Schmid22Fast,
    author={L. {Schmid} and C. {Ni} and Y. {Zhong} and and R. {Siegwart} and O. {Andersson}},
    journal={IEEE Robotics and Automation Letters},
    title={Fast and Compute-efficient Sampling-based Local Exploration Planning via Distribution Learning},
    year={2022},
    doi={10.1109/LRA.2022.3186511},
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

* Install other dependencies: 
  ```bash
  pip install -r requirements.txt
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
  export PYTHONPATH="${MY_CVAE_ROOT}/:${PYTHONPATH}"
  ```

* You are now ready to go!

## Data Repository

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

## Training the CVAE model
To train the original CVAE model, make sure you [downloaded the CVAE dataset](Data-Repository), then run:

```
cd learning 
python train_cvae.py
```

The model will start training, and periodically save the intermediate model in `learning/policies/<start_time>` and training information in `learning/runs/<start_time>`.

See `learning/config_cvae.yaml` for tunable parameters. For example, to jointly train the gain estimator (+GJ in the paper), set `x_dim` to 4 and the last dimension of the network output will be the predicted gain. 

## Training the CNN model

To train the CNN based gain estimator of the two-stage model, make sure you [downloaded the CNN dataset](Data-Repository), then run:
```
cd learning 
python train_cnn.py 
```

The model will start training and write intermediate and final models as well as a performance evaluation to `learning/runs/<start_time>`.

See `learning/config_cnn.yaml` for tunable parameters.

## Evaluating a Planner
We provide a script to run and evaluate any planner. First, set the worlds, planners, numbers of runs, numbers of sampels, and other experiment details in `experiments/config.yaml`. Then conduct the experiments by running:

```
cd experiments
python evaluate.py
```

This will run the planners in the simulator for all specified experiments and store the results in `epxeriments/results/<start_time>`. Afterwards the stored data is evaluated and exploration progress curves are plotted.

![Example_performance](https://user-images.githubusercontent.com/36043993/176662115-fb589d43-242c-4712-8308-962a0d98dae2.png)

Example performance on the demo_maze for N=5.


## Using the simulator
The simulator used to generate data and evaluate the approaches is a fully functional 2D exploration simulator. We provide a demo showcasing how some of the main features of the simulator can be used and visualized. Start the demo by running:

```
cd simulator
python demo.py
```

The demo will first display the complete randomly generated world and start pose. Then enter '1' in the terminal to let the robot explore the simulated world using a bseline planner. If it gets stuck in a local minimum a global planner will reset it.

![Simulator](https://user-images.githubusercontent.com/36043993/176603370-3dd3727d-7f65-4e35-80ba-f8419fd58516.png)

Randomly generated world (left) and robot moving around (right, orange to red pose arrows).

To generate different world files for experiments, run the world explorer:
```
cd experiments
python explore_worlds.py
```

# Additional Information
## Project Overview
The files in this repository are structured as follows:

```bash
├── data  # ---------------- # Local directory for training data.
│   ├── CNN_dataset.npy      # Provided downloadable data.
│   └── CVAE_dataset.npy
├── experiments  # --------- # Package to run experiments.
│   ├── config.yaml          # Which experiments to run.
│   ├── evaluate.py          # Run and evaluate planners.
│   ├── explore_worlds.py    # Interactively create new world files.
│   ├── models               # Provided pre-trained models.
│   ├── results              # Local directory for experiment outputs.
│   └── worlds               # Local directory to store wo
├── learning  # ------------ # Package to define and train models.
│   ├── config_cnn.yaml      # Configurations to train CVAE/CNN models.
│   ├── config_cvae.yaml
│   ├── data.py              # Data processing tools.
│   ├── model.py             # Network model definitions.
│   ├── runs                 # Local directory for training output.
│   ├── train_cnn.py         # Scripts to train the CVAE/CNN models.
│   ├── train_cvae.py
│   └── util.py              # Utility tools for networks.
├── planning  # ------------ # Package that contains all planners.
│   ├── baseline_nbvp.py     # Python implementation of RH-NBVP.
│   ├── baseline_planner.py  # Uniform sampling-based local planner.
│   ├── global_planner.py    # Frontier-based global planner.
│   ├── policy_planner.py    # Local planners using our models.
│   └── rrt_star.py          # RRT* for global path verification.
└── simulator  # ----------- # Package that contains the simulator.
    ├── config.py            # Config definition for entire simulator.
    ├── demo.py              # Example on how to use some interfaces.
    ├── robot.py             # Code for capabilities of the robot.
    ├── simulator.py         # Main interface combining all components.
    └── world.py             # Procedural world generation.
```
