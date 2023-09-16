# Volitional activation of remote place representations with a hippocampal brain‐machine interface

## Overview

Link to the paper: (To be updated)

This dataset delves into the capability of animals to activate remote place representations within the hippocampus, even when they aren't physically present in those locations. Such remote activations serve as a fundamental capability underpinning reasoning, mental simulation and imagination. By employing a hippocampal map-based brain-machine interface (BMI), we designed two specific tasks to test whether animals can intentionally control their hippocampal activity in a flexible, goal-directed, and model-based manner. Our result shows that they can perform both tasks in real-time and in single trials. This dataset provides the neural data, behavior data of these two tasks. 

## Dataset Contents:

- **Animals**: The dataset includes data from three rats.

- **Tasks**: Each of the three rats performed three tasks—Running, Jumper, and Jedi—in the same 1m x 1m virtual reality arena.
  - **Running Task**: This foundational task captures rats navigating various locations in a virtual arena. Using split training and test set, real-time decoders were trained used in subsequent tasks. The pre-trained model is provided in the dataset.
  - **Jumper Task**: Here, rats navigate to goal locations using BMI-based teleportation. The dataset provides details on how the rats' neural activity and real-time decoded locations. 
  - **Jedi Task**: Here, rats control the location of a virtual object in the arena using their hippocampal activity. The dataset provides details on how the rats' neural activity and real-time decoded locations.

 - **Data Folder Structure:**  The dataset contains three subfolders:
    - **1. Running**: This folder has two subfolders:
      - **run_before_jumper**: Contains data files for the Running task performed before the Jumper task.
      - **run_before_jedi**: Contains data files for the Running task performed before the Jedi task.
    - **2. Jumper**: Contains data files for the Jumper task.
    - **2. Jedi**: Contains data files for the Jedi task.

- **Data Structure**:
  - **Time Stamps, Population Vectors (PV)**: For each animal and task, time stamps and PVs are provided (for Jumper and Jedi tasks). The Numpy data files labels `PV_t.npy` contain the time stamps and labeled `PV.npy` contain the PVs. Pv is the population vector of 100 ms bin, which is a numpy array has shape of (number of samples, number of neurons). The time stamps are the time stamps of the PVs, which is a numpy array has shape of (number of samples, ). 
  - **real-time decoded position data during Jumper and Jedi**: For each animal participating in the BMI tasks, real-time decoded position is provided. These are stored in Numpy data files labeled `dec_pos.npy`. The position of either the remote object (in Jedi) or the rat itself (in Jumper) are causally smoothed version of the decoded location. The decoder uses `B_bins` of PV as input. Therefore, the first decoded position will start from the `B_bins` bin. The length of the decoded position vector is `B_bins-1` shorter than length of PV vectors.

    pv[0] | pv[1]| ... | pv[B_bins-2] | pv[B_bins-1] | ... | pv[n-1]
    --- | ---| --- | --- | --- | --- | ---
    N/A | N/A | ... | N/A | dec_pos[0] | ... | dec_pos[n-1-(B_bins-1)]
  - **Animal position data during Running**: For each animal during the Running task, the position data is provided. The numpy data file contains `pos.npy` are position file. 
  - **place units and place field**: For each animal and task, not all units detected in real-time were used for decoding. The place units id and place field of all units are provided. (check paper for more details). The numpy data file contains `place_units_id.npy` are place units id file (i.e., the unit index that we picked as place units). The numpy data file contains `place_fields.npy` are place field file for all units.
  - **Pre-trained deep neural network (dnn) model**: The dataset includes tools for loading the pre-trained model and performing offline decoding. Functions like `load_model` and `load_pretrained_model` in the `utils.model` module demonstrate how to do this. The weights, biases, and batch normalization (BN) layer states are stored in the ./pretrained_model folder as `.pth` files. The `.pth` file is loaded as a standard serialized pytorch state dictionary commonly used to save and restore the state of a PyTorch deep neural network model:
    ```
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load('./xxxx_model.pth'))
    ```
  - **dnn decoder code for training with data augmentation**:
  Code for researcher to modify and train their own dnn decoder for their own research purposes.  Check `training_dnn.ipynb` for more details. 
   - **Trials Data**: Detailed single trial data in the Running, Jumper, and Jedi tasks. For each trial, it includes time stamps (`t`), positions (`pos`), goal locations (`goal`), and other relevant behavior variables. 


## Code/Software/Hardware

To use the data, you will need to use the following github repository:
(TODO)

CUDA version:  11.0 

Python package dependency for this dataset: 
- numpy: 1.22.2
- scipy: 1.8.1
- sklearn: 0.23.2
- matplotlib: 3.3.4
- seaborn: 0.11.0
- pandas: 1.4.1
- pytorch: 1.11.0

Hardware:
A NVIDIA GPU is required to run the pre-trained model for offline decoding or to train your own model.


## Data Access and Utilization Guide

For each animal and task, time stamps, population vectors (PV), relevant behavior variables and decoded locations are provided. All data is stored in the numpy array format, facilitating easy loading. Also provided are the Deep neural network (DNN) decoder code and pre-trained model (Pytorch). 

Below is an example of how to load and utilize the data for rat1 during the Jedi task:

### 1. load neural data and position data (population vector of 100 ms bin)
For Jumper tasks:
```python
# pv and pv_t has the same number of samples
pv = np.load('XXXX_Jumper_PV.npy')  # PV of 100 ms bin
pv_t = np.load('XXXX_Jumper_PV_t.npy')  # time stamps of PV
dec_pos = np.load('XXXX_Jumper_dec_pos.npy')  # decoded mental position
ani_pos = np.load('XXXX_Jumper_ani_pos.npy')  # animal position in Jumper
```

For Jedi tasks:
```python
# pv and pv_t has the same number of samples
pv = np.load('XXXX_JEDI_PV.npy')  # PV of 100 ms bin
pv_t = np.load('XXXX_JEDI_PV_t.npy')  # time stamps of PV
dec_pos = np.load('XXXX_JEDI_dec_pos.npy')  # decoded mental position
ani_pos = np.load('XXXX_JEDI_obj_pos.npy')  # remote object position in JEDI
```

### 2. load pre-trained deepnet model and perform offline decoding (see document in the code for choosing parameters)

```python
model, neuron_idx, B_bins = load_pretrained_model(rat='rat1', task='Jumper')
# pv:(n_samples, n_neurons) ---> X: (n_samples, B_bins, n_neurons)
X = pv_2_spv(pv, B_bins=B_bins)
redec_pos = decode(model, X, neuron_idx, cuda=True, mimic_realtime=False) 
```

### 3. load running, jumper and jedi trials
- Running task (before Jumper or Jedi, i.e. pre-BMI during which we  collected CA1 neural data and positions for building the decoder):
```python
rat_preBMI = torch.load('./data/Running/wr112_0905_preJumper_dict')
```
rat_preBMI is a python dictionary that contains all trials in this Running task (before Jumper as the loaded file name contains `preJumper_dict`). To access a trial, use rat1_preBMI[trial_no]. Each trial is a python dictionary that contains ['t', 'pos', 'goal'] for time stamps, position and goal location, respectively. 

- Jumper task (animal control CA1 activity to move itself to the goal location, with real-time decoding):
```python
jumper_trials = torch.load('./data/Jumper/jumper_trials.pt')
```
`jumper_trials` is a python dictionary is used to store all trials in this Jumper task, in which each of three animals has its own naming `wr112`, `wr118`, `wr121` for rat1,2,3 respectively. To access data for rat1, trial 0, use `jumper_trials['wr112'][0]`, trial 1, use `jumper_trials['wr112'][1]`, etc.


- Jedi task (animal control CA1 activity to move the object to the goal location, with real-time decoding):
```python
jedi_trials =  torch.load('./data/Jedi/jedi_trials.pd')
```
`jedi_trials` is a python dictionary that contains all trials in this Jedi task, similar to `jumper_trials` data structure.


## Sharing/Access information

This is a section for linking to other ways to access the data, and for linking to sources the data is derived from, if any.

Links to other publicly accessible locations of the data:
  * 

Data was derived from the following sources:
  * 


## License (TODO)
Please cite this paper (TODO) if you use this dataset
