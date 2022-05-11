# Pho-SC--CTC
Code related to data augmentation, Pho(SC)Net, PhosNet, PhocNet is available here : https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition  
Dataset : https://drive.google.com/drive/folders/10aAPJtbsR0M1iryFXJAKyyxvYXSZfatt?usp=sharing

# Description of files inside /src/

dataloader_iam.py : This file contains code to load train/test/validation data from the given directory.  
model_phosc_cnn.py : contains code to build and train the Vanilla-CTC model, this file needs to be renamed to "model.py" so that it can be called from main.py.  
model_load_weights.py : contains code to build and train the Pho(sc)-CTC model, this file needs to be renamed to "model.py" so that it can be called from main.py.  
main.py : This is the file which binds all the other files together.
