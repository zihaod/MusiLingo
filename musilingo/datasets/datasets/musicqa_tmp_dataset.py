from musilingo.utils.lib import *
from musilingo.datasets.datasets.base_dataset import BaseDataset
from musilingo.datasets.datasets.audio_utils import download_clip

from datasets import load_dataset, load_from_disk
import os
import torch
import numpy as np
import json
import torchaudio.transforms as T

class MusicQADataset(BaseDataset):
    def __init__(self, processor, data_dir, split):
        super().__init__()
        self.split = split
        self.data_dir = os.path.join(data_dir, 'MusicQA', 'musicqa_'+split) # split is in {musiccaps_pretraining, mtt_finetuning, mtg_evaluation}
        self.resample_rate = processor.sampling_rate #music_data
        self.processor = processor
        self.subset = split.split('_')[0]
        
        with open(os.path.join(self.data_dir, self.subset+'_ann', 'train.json'), 'r') as f:
            self.ann = json.load(f)
        self.ann = self.ann['ann']

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        id = self.ann[idx]['Music']['path'][:-4]
        sampling_rate = self.ann[idx]['Music']['sampling_rate']
 
        npy_path = os.path.join(self.data_dir, self.subset+'_audio', f'{id}.npy')
        audio_array = np.load(npy_path)

        resampler = T.Resample(sampling_rate, self.resample_rate)
        audio_input = resampler(torch.from_numpy(audio_array).float())

        audio = self.processor(audio_input, 
                               sampling_rate=self.resample_rate, 
                               return_tensors="pt")['input_values'][0]
        instruction = txt = [self.ann[idx]['Question']]
        txt = [self.ann[idx]['Answer']]
        
        return {'audio': audio, 'text_input': txt, 'instruction_input': instruction}

    def collater(self, samples):
        #padding to max length in a batch
        audios = [s['audio'] for s in samples]
        audio_sizes = [len(s['audio']) for s in samples]
        audio_size = max(audio_sizes)
        txts = [" ".join(s['text_input']) for s in samples]
        instructions = [" ".join(s['instruction_input']) for s in samples]

        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        attn_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(True)
        )

        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            else: #diff < 0
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                attn_mask[i, diff:] = False

        attn_mask = attn_mask.int()

        return {'audio': collated_audios, 'text_input': txts, 'instruction_input': instructions, 'attention_mask': attn_mask}

