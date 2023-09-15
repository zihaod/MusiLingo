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
### 1. LP-MusicCaps-MSD
### 2. MusicCaps 
First create a directory named ```music_data/```, and put the processed data ```MusicCaps_audio/``` and ```MusicCaps_ann/``` into the directory. Then go to ```muvi/configs/datasets/musiccaps/default.yaml``` and set ```data_dir``` to be ```PATH/TO/music_data/```.
### 3. MusicInstruct (MI)
### 4. MusicQA


## Model Preparation
### Vicuna
You need to prepare the pretrained Vicuna weights following instructions [here](PrepareVicuna.md). Once you have the weights, create a folder named ```Vicuna_7B``` and put the weights under this folder. The final contents should look like this:
```
Vicuna_7B
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
```
Currently we use Vicuna 7B by default. Finally, go to ```musilingo/configs/models/musilingo.yaml``` and set ```llama_model``` to be ```PATH/TO/Vicuna_7B/```.

## Training
### 1. Pretraining with LP-MusicCaps-MSD
Run the following command to pretrain the model. Set ```NUM_GPU``` to be the actual available number of gpus. 
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/muvi_stage1_pretrain.yaml
```
### 2. Instruction Finetuing
For each dataset, run the command provided in the corresponding section. Again, set ```NUM_GPU``` to be the actual available number of gpus. 
#### 2.1 MusicCaps
We can use instruction tuning on MusicCaps to perform captioning tasks by giving a default question prompt.
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage2_finetune_musiccaps.yaml
```
#### 2.2 MusicInstruct 
We can run instruction tuning on the whole MI dataset, or only on either the long or the short questions.
For the whole MI dataset:
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage2_finetune_cmi.yaml
```
For the short question version:
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage2_finetune_cmi_short.yaml
```
For the long question version:
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage2_finetune_cmi_long.yaml
```
#### 2.3 MusicQA 
Run the following command to finetuning on MusicQA:
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage2_finetune_musicqa.yaml
```
