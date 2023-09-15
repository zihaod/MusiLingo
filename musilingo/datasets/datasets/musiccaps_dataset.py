from musilingo.utils.lib import *
from musilingo.datasets.datasets.base_dataset import BaseDataset
from musilingo.datasets.datasets.audio_utils import download_clip

from datasets import load_dataset, Audio
import os
import torch
import numpy as np
import json
import torchaudio.transforms as T

class MusicCapsDataset(BaseDataset):
    def __init__(self, processor, data_dir, split):
        super().__init__()
        self.split = split # train/test
        self.data_dir = data_dir # music_data
        self.resample_rate = processor.sampling_rate
        self.processor = processor
        
        ann_path = os.path.join(data_dir, 'MusicCaps_ann', 'all.json')
        with open(ann_path, 'r') as f:
            self.all_ann = json.load(f)
        self.all_ann = self.all_ann['ann']
        
        # train/test split
        self.ann = []
        is_eval = self.split == 'test'
        for item in self.all_ann:
            if item['is_audioset_eval'] == is_eval:
                self.ann.append(item)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ytid = self.ann[idx]['ytid']
        sampling_rate = self.ann[idx]['sampling_rate']
        npy_path = os.path.join(self.data_dir, 'MusicCaps_audio', f'{ytid}.npy')
        raw_audio = np.load(npy_path)
        
        resampler = T.Resample(sampling_rate, self.resample_rate)
        audio_input = resampler(torch.from_numpy(raw_audio).float())
        
        audio = self.processor(audio_input, 
                               sampling_rate=self.resample_rate, 
                               return_tensors="pt")['input_values'][0]
        txt = [self.ann[idx]['caption']]
        instruction = ['Please give a caption to the music.']

        return {'audio': audio, 'text_input': txt, 'instruction_input': instruction,  'ytid': ytid}

    
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

class MusicCapsDatasetWithExtract(BaseDataset):
    def __init__(self, processor, data_dir, split):
        super().__init__()
        self.split = split
        self.ds = load_dataset('google/MusicCaps', split=split)
        self.data_dir = data_dir
        self.resample_rate = processor.sampling_rate
        self.processor = processor
        
        def process_data(example):
            outfile_path = os.path.join(data_dir, f"{example['ytid']}.wav")
            status = True
            if not os.path.exists(outfile_path):
                status = False
                status, log = download_clip(
                    example['ytid'],
                    outfile_path,
                    example['start_s'],
                    example['end_s'],
                )

            example['audio'] = outfile_path
            example['download_status'] = status
            return example
        
        self.ds = self.ds.map(
                              process_data,
                              keep_in_memory=False
                             ).cast_column('audio', Audio(sampling_rate=self.resample_rate))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            raw_audio = self.ds[idx]['audio']['array']
        except:
            print("missing data point")
            return None
        audio = self.processor(raw_audio, 
                                     sampling_rate=self.resample_rate, 
                                     return_tensors="pt")['input_values'][0]
        txt = [self.ds[idx]['caption']]

        return {'audio': audio, 'text_input': txt}

    def collater(self, samples):
        samples = [s for s in samples if s]
        audios = [s['audio'] for s in samples]
        audio_sizes = [len(s['audio']) for s in samples]
        audio_size = max(audio_sizes)
        txts = [" ".join(s['text_input']) for s in samples]

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

        return {'audio': collated_audios, 'text_input': txts, 'attention_mask': attn_mask}

    

    
