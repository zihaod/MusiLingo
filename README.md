# MusiLingo
This repo contains the code for the following paper. 
__[MusiLingo: Bridging Music and Text with Pre-trained Language Models for Music Captioning and Query Response](https://arxiv.org/abs/2309.08730)__

You can also refer to our [Huggingface collection](https://huggingface.co/collections/m-a-p/musilingo-660ea663da171832d5722c51) for quick start.

## Environment setup
To get started, git clone the repo and install the required dependencies using the following commands:
```
git clone https://github.com/zihaod/MusiLingo
cd MusiLingo
conda create -n musilingo python=3.10.6
conda activate musilingo
pip install -r requirements.txt
```


## Data Preparation
### 1. LP-MusicCaps-MSD
The MusiLingo model is pre-trained on LP-MusicCaps-MSD dataset and we only provide the annotation of the dataset in this repo under ```data/music_data/msd`````. The audio is part of the Million Song Dataset (MSD) which you may not be able to download from Internet easily.
### 2. MusicCaps 
We provide a copy of annotation of MusicCaps dataset under ```data/music_data/MusicCaps_ann```. The audio is part of Google's AudioSet and you can download it from YouTube.
### 3. MusicInstruct (MI)
We develop MI dataset and saved under ```data/music_data/MusicInstruct```. The audios are identical with MusicCaps. You can also find more information at the [Huggingface page](https://huggingface.co/datasets/m-a-p/Music-Instruct/tree/main).
### 4. MusicQA
You can doanloaw the MusicQA dataset to ```data/music_data/MusicQA``` from Huggingface.


## Model Preparation
### Vicuna
You need to prepare the pretrained Vicuna weights following instructions [here](PrepareVicuna.md). Once you have the weights, put the weights under the ```model/7B_vicuna``` folder. The final contents should look like this:
```
Vicuna_7B
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
```
Currently we use Vicuna 7B v0 by default. Finally, go to ```musilingo/configs/models/musilingo.yaml``` and set ```llama_model``` to be ```PATH/TO/Vicuna_7B/```. Alternatively, you can also set it to the huggingface path ```lmsys/vicuna-7b-delta-v0```.
### MERT
We use MERT-v1-330M as music encoder for MusiLingo model. You can download it from [Huggingface page](https://huggingface.co/m-a-p/MERT-v1-330M) to ```model/MERT-v1-330M```

## Training
### 1. Pretraining with LP-MusicCaps-MSD
Run the following command to pretrain the model. Set ```NUM_GPU``` to be the actual available number of gpus. 
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage1_pretrain.yaml
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
Run the following command to finetune on MusicQA:
```
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/musilingo_stage2_finetune_musicqa.yaml
```

## Inference
To do the inference on MusicInstruct dataset, use the following code
```Python qa.py --qa_type short```
```Python qa.py --qa_type long```
### Model Checkpoints

If you cannot download from the ckpt in this repo, you can download the pretrained model checkpoints


| MusiLingo (long) | MusiLingo (MusicQA) | MusiLingo (short)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1FtPRHVL3w0CYMTmR2Bpju9knYKIPMlzC/view?usp=drive_link) |[Download](https://drive.google.com/file/d/1-jK5PKU0ZCNIu5F7JAqr5S-Ei_urgYei/view?usp=drive_link) | [Download](https://drive.google.com/file/d/16LFAK3dM2a3xlU3SvgToa3DboImndjLU/view?usp=drive_link) |
| ```model/ckpt/long/``` | ```model/ckpt/musicqa/``` | ```model/ckpt/short/``` |

## Citing This Work

If you find the work useful for your research, please consider citing it using the following BibTeX entry:
```
@inproceedings{deng2024musilingo,
  title={MusiLingo: Bridging Music and Text with Pre-trained Language Models for Music Captioning and Query Response},
  author={Deng, Zihao and Ma, Yinghao and Liu, Yudong and Guo, Rongchen and Zhang, Ge and Chen, Wenhu and Huang, Wenhao and Benetos, Emmanouil},
  booktitle={Proceedings of the 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)},
  year={2024},
  organization={Association for Computational Linguistics}
}
```