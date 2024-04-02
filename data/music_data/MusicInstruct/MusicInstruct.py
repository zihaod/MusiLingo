import os
import  json
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers import Wav2Vec2FeatureExtractor


class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)




class CMIDataset(BaseDataset):
    def __init__(self, processor, data_dir, split, question_type='all'):
        super().__init__()
        self.split = split # train or test
        self.data_dir = data_dir #music_data
        self.resample_rate = processor.sampling_rate
        self.processor = processor

        ann_path = os.path.join(data_dir, 'MusicInstruct/music_instruct.json')
        if question_type == 'short':
            ann_path = os.path.join(data_dir, 'MusicInstruct/music_instruct_short.json')
        elif question_type == 'long':
            ann_path = os.path.join(data_dir, 'MusicInstruct/music_instruct_long.json')
        
        with open(ann_path, 'r') as f:
            self.all_ann = json.load(f)
        self.all_ann = self.all_ann['QA']
        
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
        
        instruction = [self.ann[idx]['question']]
        txt = [self.ann[idx]['answer']]
        
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

if __name__ == '__main__':
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)

    ds_short = CMIDataset(processor, '/content/drive/MyDrive/music_data', split='test', question_type='short')
    ds_long = CMIDataset(processor, '/content/drive/MyDrive/music_data', split='test', question_type='long')
    ds_mixed = CMIDataset(processor, '/content/drive/MyDrive/music_data', split='test', question_type='all')
