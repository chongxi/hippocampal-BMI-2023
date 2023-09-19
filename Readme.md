# Volitional activation of remote place representations with a hippocampal brain‐machine interface

## Overview

This repository is associated with the following paper:
**Lai C, Tanaka S, Harris TD, Lee AK. Volitional activation of remote place representations with a hippocampal brain‐machine interface. Science, 2023 (in press).**

This dataset demonstrates the ability of animals to activate remote place representations within the hippocampus when they aren't physically present at those locations. Such remote activations serve as a fundamental capability underpinning memory recall, mental simulation/planning, imagination, and reasoning. By employing a hippocampal map-based brain-machine interface (BMI), we designed two specific tasks to test whether animals can intentionally control their hippocampal activity in a flexible, goal-directed, and model-based manner. Our results show that animals can perform both tasks in real-time and in single trials. This dataset provides the neural and behavior data of these two tasks. The details of the tasks and results are described in the paper. 

## Dataset and pre-trained model access:

Download `data.7z` and `model.7z` from https://doi.org/10.5281/zenodo.8360872

- Unzip the `data.7z` to get a `data` folder. The `data` folder contains three subfolders:
    - **1. Running**: This folder has two subfolders:
      - **run_before_jumper**: Contains data files for the Running task performed before the Jumper task.
      - **run_before_jedi**: Contains data files for the Running task performed before the Jedi task.
    - **2. Jumper**: Contains data files for the Jumper task.
    - **2. Jedi**: Contains data files for the Jedi task.

- Unzip the `model.7z` to get a `pretrained_model` folder, which contains all 6 pretrained models (`pth` files) trained using the data from the `Running` tasks, 3 used in `Jumper` tasks and 3 used in the `Jedi` tasks. 

Put the `data` folder and `pretrained_model` folder in the same directory as the code in this repository.

## Dataset Contents:

- **Animals**: The dataset includes data from three rats.

- **Tasks**: Each of the three rats performed three tasks—Running, Jumper, and Jedi—in the same 1m x 1m virtual reality arena.
  - **Running Task**: In this task, rats navigate around the virtual arena. The data were split into training and test sets, allowing us to train real-time decoders of the animal's current location for subsequent BMI tasks. The pre-trained model is provided in the dataset.
  - **Jumper Task**: Here, rats navigate to goal locations using BMI-based teleportation using their hippocampal activity. The dataset provides the rats' neural activity and real-time decoded locations. 
  - **Jedi Task**: Here, rats control the location of a virtual object in the arena using their hippocampal activity. The dataset provides the rats' neural activity and real-time decoded locations. 


- **Data Structure**:
  - **Time Stamps, Population Vectors (PV)**: For each animal and task, time stamps and PVs are provided. The Numpy data files labeled `PV_t.npy` contain the time stamps and labeled `PV.npy` contain the PVs. PV is the population vector using 100 ms bin, which is a numpy array with shape (number of samples, number of neurons). The time stamps are the time stamps of the PVs, which is a numpy array with shape (number of samples, ). 

  - **Animal location data during Running task**: For each animal during the Running task, the location data is provided. The numpy data file contains `pos.npy` are location file. 

  - **Real-time decoded location data during Jumper and Jedi tasks**: For each animal and BMI task, real-time decoded position is provided. These are stored in Numpy data files labeled `dec_pos.npy`. The locations of either the rat itself (in Jumper) or the remote object (in Jedi) are causally smoothed versions of the decoded locations. The decoder uses `B_bins` of PV as input. Therefore, the first decoded location will start at the `B_bins`-th bin. The number of the decoded locations is `B_bins-1` less than the number of PVs.

    pv[0] | pv[1]| ... | pv[B_bins-2] | pv[B_bins-1] | ... | pv[n-1]
    --- | ---| --- | --- | --- | --- | ---
    N/A | N/A | ... | N/A | dec_pos[0] | ... | dec_pos[n-1-(B_bins-1)]

  - **Trials data**: Detailed single trial data for the Running, Jumper, and Jedi tasks. For each trial, it includes time stamps (`t`), locations (`pos`), goal locations (`goal`), and other relevant behavioral variables.

  - **Place units and place fields**: For each animal and task, not all units detected in real-time were used for decoding. The place unit ids and place fields of all units are provided (see paper for more details). The numpy data file `place_units_id.npy` is the place unit id file (i.e., the ids of the units that we selected as place units). The numpy data file `place_fields.npy` contains the place fields for all units. 

  - **Pre-trained deep neural network (DNN) model**: The dataset includes tools for loading the pre-trained decoder and performing offline decoding. Functions like `load_model` and `load_pretrained_model` in the `utils.model` module demonstrate how to do this. The weights, biases, and batch normalization (BN) layer states are stored in the `./pretrained_model` folder as `.pth` files. The `.pth` file is loaded as a standard serialized pytorch state dictionary commonly used to save and restore the state of a PyTorch deep neural network model:
    ```
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load('./xxxx_model.pth'))
    ```
  - **DNN decoder code for training with data augmentation**:
  Code for researchers to modify and train their own DNN decoder for their own research purposes. See `training_dnn.ipynb` for more details.


## Code/Software/Hardware

The versions we use:

CUDA version:  11.0 
Python: 3.8.5
Python package dependency for this dataset: 
- numpy: 1.22.2
- scipy: 1.8.1
- sklearn: 0.23.2
- matplotlib: 3.3.4
- seaborn: 0.11.0
- pandas: 1.4.1
- pytorch: 1.11.0

Hardware:
An NVIDIA GPU is required to run the pre-trained model for offline decoding or to train your own model.


## Utilization Guide

For each animal and task, time stamps, population vectors (PV), and actual or decoded locations are provided. All data is stored in the numpy array format. The Deep neural network (DNN) decoder code and pre-trained model are also provided (Pytorch). 

Below is an example of how to load and utilize the data for `rat1` during the Jedi task:

### 1. load neural data and location data (PVs in 100 ms bins)
Running tasks:
```python
train_PV = np.load('XXXX_train_FA40_PV.npy')
train_pos = np.load('XXXX_train_pos.npy')
test_PV = np.load('XXXX_test_FA40_PV.npy')
test_pos = np.load('XXXX_test_pos.npy')
```

Jumper tasks:
```python
# pv and pv_t has the same number of samples
pv = np.load('XXXX_Jumper_PV.npy')  # PV of 100 ms bin
pv_t = np.load('XXXX_Jumper_PV_t.npy')  # time stamps of PV
dec_pos = np.load('XXXX_Jumper_dec_pos.npy')  # decoded mental position
ani_pos = np.load('XXXX_Jumper_ani_pos.npy')  # animal position in Jumper
```

Jedi tasks:
```python
# pv and pv_t has the same number of samples
pv = np.load('XXXX_JEDI_PV.npy')  # PV of 100 ms bin
pv_t = np.load('XXXX_JEDI_PV_t.npy')  # time stamps of PV
dec_pos = np.load('XXXX_JEDI_dec_pos.npy')  # decoded mental position
ani_pos = np.load('XXXX_JEDI_obj_pos.npy')  # remote object position in JEDI
```

### 2. load pre-trained DNN model and perform offline decoding (see document in the code for choosing parameters)

```python
model, neuron_idx, B_bins = load_pretrained_model(rat='rat1', task='Jumper')
# pv:(n_samples, n_neurons) ---> X: (n_samples, B_bins, n_neurons)
X = pv_2_spv(pv, B_bins=B_bins)
redec_pos = decode(model, X, neuron_idx, cuda=True, mimic_realtime=False) 
```

### 3. load Running, Jumper and Jedi trials
- Running task (before Jumper or Jedi tasks, i.e., pre-BMI during which we collected CA1 neural data and locations for building the decoder):
```python
running_trials = torch.load('./data/Running/wr112_0905_preJumper_dict')
```
`running_trials` is a python dictionary that contains all trials in this Running task. To access a trial, use rat1_preBMI[trial_no]. Each trial is a python dictionary that contains ['t', 'pos', 'goal'] for time stamps, animal locations, and goal location for that trial, respectively. 

- Jumper task (animal controls its hippocampal CA1 activity to move itself to the goal location with real-time decoding):
```python
jumper_trials = torch.load('./data/Jumper/jumper_trials.pt')
```
`jumper_trials` is a python dictionary used to store all trials in Jumper tasks, in which each of three animals has its own name `wr112`, `wr118`, `wr121` for rat1,2,3 respectively. To access data for rat1, trial 0, use `jumper_trials['wr112'][0]`, trial 1, use `jumper_trials['wr112'][1]`, etc.

- Jedi task (animal controls its hippocampal CA1 activity to move the object to the goal location with real-time decoding):
```python
jedi_trials =  torch.load('./data/Jedi/jedi_trials.pd')
```
`jedi_trials` is a python dictionary that contains all trials in this Jedi task, similar to `jumper_trials` data structure.


## License: CC BY 4.0
If you use this dataset, please cite this paper:
**Lai C, Tanaka S, Harris TD, Lee AK. Volitional activation of remote place representations with a hippocampal brain‐machine interface. Science, 2023 (in press).**
