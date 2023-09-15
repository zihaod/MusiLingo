# MusiLingo
This repo contains the code for the MusiLingo model.

## Environment setup
To get started, git clone the repo and install the required dependencies using the following commands:
```
git clone https://github.com/zihaod/MusiLingo
cd MusiLingo
conda create -n musilingo python=3.10.6
conda activate musilingo
pip install requirements.txt
```


## Data Preparation
### 1. MusicCaps 
First create a directory named ```music_data/```, and put the processed data ```MusicCaps_audio/``` and ```MusicCaps_ann/``` into the directory. Then go to ```muvi/configs/datasets/musiccaps/default.yaml``` and set ```data_dir``` to be ```PATH/TO/music_data/```.
### 2. LP-MusicCaps
### 3. MusicQA
### 4. MusicInstruct

## Model Preparation
### Vicuna
You need to prepare the pretrained Vicuna weights following instructions [here](PrepareVicuna.md). Once you have the weights, create a folder named ```vicuna_weights``` and put the weights under this folder. The final contents should look like this:
```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
```
Note that currently we use Vicuna 7B, but the weights could come from a larger version. Finally, go to ```muvi/configs/models/muvi.yaml``` and set ```llama_model``` to be ```PATH/TO/vicuna_weights/```.

## Training
### 1. First stage pretraining
Run the following command to pretrain the model. Set ```NUM_GPU``` to be the actual available number of gpus on your node. 
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/muvi_stage1_pretrain.yaml
```
