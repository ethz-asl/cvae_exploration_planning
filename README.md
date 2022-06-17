# cvae_exploration_planning
**CVAE Exploration Planning** is a package providing an open-source implementation of the simulator, networks, and planners presented in our paper. TODO.


# Table of Contents
**Credits**
* [Paper and Video](#Paper-and-Video)

**Setup**
* [Packages](#Packages)
* [Dependencies](#Dependencies)
* [Installation](#Installation)
* [Data Repository](#Data-Repository)

**Examples**
* [Configuring a Planner](#Configuring-a-Planner)
* [Run an Experiment](#Run-an-Experiment)

For additional information please see the [wiki](https://github.com/ethz-asl/mav_active_3d_planning/wiki).

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
TODO add all install instructions here and test them.
Download our dataset at [dataset]()

# Examples
TODO: conda environment reproduce.

Add the project folder to the python environment

```
export PYTHONPATH='/path/to/this/repo'
```
cd
To train the CVAE model:

```
cd learning/
python train.py
```

See `learning/config.yaml` for tunable parameters

To evaluate the learned model, save the model to experiments/models, and

```
python evaluate.py
```
Choose the world, planner, runs in `experiments/config.yaml`.