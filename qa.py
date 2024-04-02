import pandas as pd
import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from transformers import StoppingCriteria, StoppingCriteriaList

from musilingo.datasets.datasets.musicqa_dataset import MusicQADataset
from musilingo.datasets.datasets.cmi_dataset import CMIDataset
from musilingo.datasets.datasets.musiccaps_dataset import MusicCapsDataset
from musilingo.models.muvi_model import MusiLingo



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--qa_type', type=str, default='short')
    config = parser.parse_args()
    
    path = "/data/home/acw688/MusiLingo"

    raw_data = pd.read_csv(f'{path}/data/music_data/musiccaps-public.csv', sep=',')
    eval_set = set()
    for i in range(len(raw_data)):
        if raw_data['is_audioset_eval'][i]:
            eval_set.add(raw_data['ytid'][i])


    ann_dir = f'{path}/data/music_data/MusicInstruct'
    data_dir = f'{path}/data/music_data/MusicCaps_audio'

    with open(os.path.join(ann_dir, "music_instruct.json"), "r") as f:
        ann = json.load(f)


    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)

    if config.qa_type == "short":
        ds = CMIDataset(processor,f'{path}/data/music_data', 'test', question_type='short')
    elif config.qa_type == "long":
        ds = CMIDataset(processor,f'{path}/data/music_data', 'test', question_type='long')
    elif config.qa_type == "musicqa":
        ds = MusicQADataset(processor, f'{path}/data/music_data', 'Eval')
    elif config.qa_type == "musiccaps":
        ds = MusicCapsDataset(processor, f'{path}/data/music_data', 'test')



    dl = DataLoader(
                    ds,
                    batch_size=1,
                    num_workers=0,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=True,
                    collate_fn=ds.collater
                    )

    model = MusiLingo(llama_model=f'{path}/model/7B_vicuna',
                prompt_template='###Human: {} ###Assistant: ')

    if config.qa_type == "short":
        ckpt_path = f"{path}/model/ckpt/short/checkpoint_11.pth"
    elif config.qa_type == "long":
        ckpt_path = f"{path}/model/ckpt/long/checkpoint_5.pth"
    elif config.qa_type == "musicqa":
        ckpt_path = f"{path}/model/ckpt/musicqa/checkpoint_5.pth"
    elif config.qa_type == "musiccaps":
        pass
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()



    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops=[], encounters=1):
            super().__init__()
            self.stops = stops
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            for stop in self.stops:
                if torch.all((stop == input_ids[0][-len(stop):])).item():
                    return True
            return False



    def answer(self, samples, stopping, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.5,
            repetition_penalty=1.0, length_penalty=1, temperature=0.1, max_length=2000):
        audio = samples["audio"].cuda()
        audio_embeds, atts_audio = self.encode_audio(audio)
        if 'instruction_input' in samples:  # instruction dataset
            #print('Instruction Batch')
            instruction_prompt = []
            for instruction in samples['instruction_input']:
                prompt = '<Audio><AudioHere></Audio> ' + instruction
                instruction_prompt.append(self.prompt_template.format(prompt))
            audio_embeds, atts_audio = self.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)
        self.llama_tokenizer.padding_side = "right"
        batch_size = audio_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=torch.long,
                        device=torch.device('cuda')) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_audio[:, :1]
        inputs_embeds = torch.cat([bos_embeds, audio_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_audio], dim=1)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # if there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text

    stopping = StoppingCriteriaList([StoppingCriteriaSub([torch.tensor([835]).cuda(),
                                    torch.tensor([2277, 29937]).cuda()])])


    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score as meteor_scorer
    from nltk.tokenize import wordpunct_tokenize
    import json
    from bert_score import score
    from tqdm.auto import tqdm

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    mult_reference = []
    candidates = []
    for idx, sample in tqdm(enumerate(dl)):
        #sample = next(iter(dl))
        ans = answer(model, sample, stopping, length_penalty=100, temperature=0.1)
        txt = sample['text_input'][0]
        print(txt)
        print(ans)
        mult_reference.append(txt)
        candidates.append(ans)


    rouge_score, bleu_score, bleu4_score, meteor_score = 0, 0, 0, 0
    for ref, cand in tqdm(zip(mult_reference, candidates), total=len(mult_reference)):
        rouge_score += scorer.score(ref, cand)['rougeL'].recall
        cand_split = wordpunct_tokenize(cand)
        ref_split = wordpunct_tokenize(ref)
        bleu4_score += sentence_bleu([ref], cand, weights=(0.0, 0.0, 0.0, 1.0))
        bleu_score += sentence_bleu([ref], cand)
        meteor_score += meteor_scorer([ref_split], cand_split)
    rouge_score, bleu_score, bleu4_score, meteor_score = rouge_score / (len(candidates)), bleu_score / (len(candidates)), bleu4_score / (len(candidates)), meteor_score / (len(candidates))
    P, R, F1 = score(candidates, mult_reference, lang="en", verbose=True)
    bert_score = R.mean().item()
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU-4 Score: {bleu4_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"BERT Score: {bert_score}")
