import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from transformers import Wav2Vec2FeatureExtractor

from musilingo.common.config import Config
from musilingo.common.dist_utils import get_rank
from musilingo.common.registry import registry
from musilingo.conversation.conversation import Chat, CONV_MUSIC

# imports modules for registration
from musilingo.datasets.builders import *
from musilingo.models import *
from musilingo.processors import *
from musilingo.runners import *
from musilingo.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna': CONV_MUSIC}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_MUSIC = conv_dict[model_config.model_type]

audio_processor_name = cfg.datasets_cfg.musicqa.processor
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_processor_name, trust_remote_code=True)
chat = Chat(model, audio_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================


def gradio_reset(chat_state, audio_list):
    if chat_state is not None:
        chat_state.messages = []
    if audio_list is not None:
        audio_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your music first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, audio_list


def upload_audio(gr_audio, text_input, chat_state):
    if gr_audio is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_MUSIC.copy()
    audio_list = []
    llm_message = chat.upload_audio(gr_audio, chat_state, audio_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, audio_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, audio_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              audio_list=audio_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, audio_list


title = """<h1 align="center">Demo of MUVI</h1>"""
description = """<h3>This is the demo of MUVI. Upload your musics and start chatting!</h3>"""
#article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
#"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    #gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            audio = gr.Audio(type="numpy") # or type="filepath"
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            audio_list = gr.State()
            chatbot = gr.Chatbot(label='MUVI')
            text_input = gr.Textbox(label='User', placeholder='Please upload your music first', interactive=False)
    
    upload_button.click(upload_audio, [audio, text_input, chat_state], [audio, text_input, upload_button, chat_state, audio_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, audio_list, num_beams, temperature], [chatbot, chat_state, audio_list]
    )
    clear.click(gradio_reset, [chat_state, audio_list], [chatbot, audio, text_input, upload_button, chat_state, audio_list], queue=False)

demo.launch(share=True, enable_queue=True)
