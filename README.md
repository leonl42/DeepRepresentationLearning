# Dataset 

Download the dataset [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and put it into ./dataset/

# Training

Training can be started with:

- python train_evaluate.py SEED 
- python train_lewis.py SEED NUMBER_OF_DISTRACTORS
- python train_autoencoder.py SEED 
- python train_predict_transforms.py SEED 

The corresponding loss/accuracy save files will be saved in a ./saves/ folder.

# Evaluation 

In order to get the final evaluation plot run all cells in final.ipynb. You might have to add or leave out cells in 
order to take into account experiments one did or did not do.