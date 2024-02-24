# NIDS
Major project on Network Intrusion Detection System

## Repository structure
- majorproj-main
  - contains images, LaTEX files and project report in pdf format
- afaiLabeling
  - contains 2/3 of the validation set we created
- micMultiBB1
  - a folder containing the saved LSTM model
- X_val_nulti_with_ping.csv and y_val_multi_with_ping.csv
  - validation datasets used in modelLoading.py/.ipynb
- for_test.ipynb
  - preprocessing the validation dataset
- preprocessBB.py
  - preprocessing the original dataset
- lstMulti.py and lstMulti.ipynb
  - code containing the LSTM model used in the project.
## Dataset
The InSDN dataset (https://aseados.ucd.ie/datasets/SDN/)

## Prerequisites
 - Keras 
 - Sklearn 
 - Pandas 
 - Numpy
 - Matplotlib

## Running the project
The project can be run on python by running the three python files serially or in Colab/Jupyter notebook. If the choice is to run the notebook, the lstMulti.ipynb file must be run in colab.
