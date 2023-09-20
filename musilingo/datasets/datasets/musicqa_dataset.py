from musilingo.utils.lib import *
from musilingo.datasets.datasets.base_dataset import BaseDataset
from musilingo.datasets.datasets.audio_utils import download_clip

from datasets import load_dataset, Audio
import torchaudio
import os
import torch
import numpy as np
import json
import random
import torchaudio.transforms as T

RESAMPLE_RATE=24000

def load_audio(
    file_path, 
    target_sr, 
    is_mono=True, 
    is_normalize=False,
    crop_to_length_in_sec=None, 
    crop_to_length_in_sample_points=None, 
    crop_randomly=False, 
    pad=False,
    return_start=False,
    device=torch.device('cpu')
):
    """Load audio file and convert to target sample rate.
    Supports cropping and padding.

    Args:
        file_path (str): path to audio file
        target_sr (int): target sample rate, if not equal to sample rate of audio file, resample to target_sr
        is_mono (bool, optional): convert to mono. Defaults to True.
        is_normalize (bool, optional): normalize to [-1, 1]. Defaults to False.
        crop_to_length_in_sec (float, optional): crop to specified length in seconds. Defaults to None.
        crop_to_length_in_sample_points (int, optional): crop to specified length in sample points. Defaults to None. Note that the crop length in sample points is calculated before resampling.
        crop_randomly (bool, optional): crop randomly. Defaults to False.
        pad (bool, optional): pad to specified length if waveform is shorter than specified length. Defaults to False.
        device (torch.device, optional): device to use for resampling. Defaults to torch.device('cpu').
    
    Returns:
        torch.Tensor: waveform of shape (1, n_sample)
    """
    # TODO: deal with target_depth
    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        waveform, sample_rate = torchaudio.backend.soundfile_backend.load(file_path)
    if waveform.shape[0] > 1:
        if is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if is_normalize:
        waveform = waveform / waveform.abs().max()
    
    waveform, start = crop_audio(
        waveform, 
        sample_rate, 
        crop_to_length_in_sec=crop_to_length_in_sec, 
        crop_to_length_in_sample_points=crop_to_length_in_sample_points, 
        crop_randomly=crop_randomly, 
        pad=pad,
    )
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = waveform.to(device)
        resampler = resampler.to(device)
        waveform = resampler(waveform)
    
    if return_start:
        return waveform, start
    return waveform


def crop_audio(
    waveform, 
    sample_rate, 
    crop_to_length_in_sec=None, 
    crop_to_length_in_sample_points=None, 
    crop_randomly=False, 
    pad=False,
):
    """Crop waveform to specified length in seconds or sample points.
    Supports random cropping and padding.

    Args:
        waveform (torch.Tensor): waveform of shape (1, n_sample)
        sample_rate (int): sample rate of waveform
        crop_to_length_in_sec (float, optional): crop to specified length in seconds. Defaults to None.
        crop_to_length_in_sample_points (int, optional): crop to specified length in sample points. Defaults to None.
        crop_randomly (bool, optional): crop randomly. Defaults to False.
        pad (bool, optional): pad to specified length if waveform is shorter than specified length. Defaults to False.

    Returns:
        torch.Tensor: cropped waveform
        int: start index of cropped waveform in original waveform
    """
    assert crop_to_length_in_sec is None or crop_to_length_in_sample_points is None, \
    "Only one of crop_to_length_in_sec and crop_to_length_in_sample_points can be specified"

    # convert crop length to sample points
    crop_duration_in_sample = None
    if crop_to_length_in_sec:
        crop_duration_in_sample = int(sample_rate * crop_to_length_in_sec)
    elif crop_to_length_in_sample_points:
        crop_duration_in_sample = crop_to_length_in_sample_points

    # crop
    start = 0
    if crop_duration_in_sample:
        if waveform.shape[-1] > crop_duration_in_sample:
            if crop_randomly:
                start = random.randint(0, waveform.shape[-1] - crop_duration_in_sample)
            waveform = waveform[..., start:start + crop_duration_in_sample]

        elif waveform.shape[-1] < crop_duration_in_sample:
            if pad:
                waveform = torch.nn.functional.pad(waveform, (0, crop_duration_in_sample - waveform.shape[-1]))
    
    return waveform, start
  

class MusicQADataset(BaseDataset):
    def __init__(self, processor, data_dir, split):
        super().__init__()
        self.split = split # Finetune/Pretrain/
        self.data_dir = data_dir #music_data
        self.resample_rate = processor.sampling_rate
        self.processor = processor

        ann_path = os.path.join(data_dir, f'MusicQA/{split}MusicQA.json')
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
                

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        data_path = self.ann[idx]['audio_name']
        filename_path = os.path.join(self.data_dir, 'MusicQA', 'audios', f'{data_path}')

        try:
            audio = load_audio(
                filename_path,
                target_sr=RESAMPLE_RATE,
                is_mono=True,
                is_normalize=False,
                # crop_to_length_in_sec=31,
                crop_to_length_in_sample_points=int(15.5*RESAMPLE_RATE),
                crop_randomly=True, 
                pad=True,
            )
            audio = audio.squeeze(0)
    
            instruction = [self.ann[idx]['conversation'][0]['value']]
            txt = [self.ann[idx]['conversation'][1]['value']]
            
            return {'audio': audio, 'text_input': txt, 'instruction_input': instruction}

        except Exception as e:
            print(f"skip audio {data_path} because of {e}")
            return self.__getitem__(10)

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
