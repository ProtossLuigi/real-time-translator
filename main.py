from pathlib import Path
import datetime
from typing import List
import requests

import gradio as gr
from openai import OpenAI
import deepl
import torch

INPUT_AUDIO_PATH = Path(__file__).parent / 'speech.wav'
OUTPUT_AUDIO_PATH = Path(__file__).parent / 'speech.mp3'
SAMPLING_RATE = 48000
DETECTION_THRESHOLD = .5
WINDOW_SIZE_SAMPLES = 100000
PATIENCE = datetime.timedelta(seconds=2)

with open('keys.txt') as fd:
    openai_key, deepl_key = fd.readlines()
    openai_key, deepl_key = openai_key[:-1], deepl_key[:-1]

client = OpenAI(api_key=openai_key)
translator = deepl.Translator(deepl_key)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

vad_iterator = VADIterator(model)
speech_data = None
last_positive = None

def get_langs(api_key: str) -> List[str]:
    response = requests.get(url='https://api-free.deepl.com/v2/languages?type=target', headers={'Authorization': f'DeepL-Auth-Key {api_key}'})
    return [lang['language'] for lang in response.json()]

def proccess(audio, voice, lang):
    """Translates voice."""
    with open(audio, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file
        )
    
    translated = translator.translate_text(transcript.text, target_lang=lang)

    response = client.audio.speech.create(
        model='tts-1',
        voice=voice,
        input=translated.text
    )
    response.stream_to_file(OUTPUT_AUDIO_PATH)

    return OUTPUT_AUDIO_PATH

def detect_voice(audio, *args):
    """Detects voice in streaming audio."""
    global speech_data, last_positive
    wav = read_audio(audio, sampling_rate=SAMPLING_RATE)
    speech_prob = model(wav, SAMPLING_RATE).item()

    if speech_prob > DETECTION_THRESHOLD:
        last_positive = datetime.datetime.now()
        if speech_data is None:
            speech_data = wav
        else:
            speech_data = torch.cat((speech_data, wav))
        raise RuntimeError()
    else:
        if speech_data is None:
            vad_iterator.reset_states()
            raise RuntimeError()
        if datetime.datetime.now() - PATIENCE > last_positive:
            speech_data = torch.cat((speech_data, wav))
            raise RuntimeError()
        save_audio(INPUT_AUDIO_PATH, speech_data, SAMPLING_RATE)
        speech_data = None
        last_positive = None
        vad_iterator.reset_states()
        return proccess(INPUT_AUDIO_PATH, *args)

def main():
    gr.Markdown('Real time voice translator.')
    with gr.Blocks() as demo:
        with gr.Row():
            voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
            voice_choice = gr.Dropdown(voices, value=voices[0], label='Voice')
            langs = get_langs(deepl_key)
            lang_choice = gr.Dropdown(langs, value='EN-US', label='Target language')

        mic = gr.Microphone(streaming=True, type='filepath', label="Audio input")
        speaker = gr.Audio(interactive=False, streaming=True, autoplay=True, visible=False)

        mic.stream(detect_voice, inputs=[mic, voice_choice, lang_choice], outputs=[speaker])

    demo.queue().launch(debug=False, show_error=False)

if __name__ == '__main__':
    main()
