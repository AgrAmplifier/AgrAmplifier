# AgrAmplifier

## Understanding the code
To run the experiments, please run *dataset*_script.py, replace *dataset* with the dataset to experiment with
The code can execute directly without downloading dataset file.

Other files in the repository
* __constants.py__ Experiment constants for utility
* __Defender.py__ Byzantine-robust aggregators
* __FL_Models.py__ The models used for experiments

To adjust the experiment to run, update the 
	for att_mode in ...
	for exp in ...
with corresponding value in __constants.py__, e.g. *constants.att_mode* / *costants.targeted_att*. Comments provided in __constants.py__

## Requirements
* Python 3.7.0 or higher
* PyTorch 1.7.1
* numpy 1.19.2
