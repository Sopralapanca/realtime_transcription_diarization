import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import threading

from datetime import datetime, timedelta
from queue import Queue
from huggingface_hub.utils import HFValidationError
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import queue
from time import sleep
import time
from sys import platform
from pydub import AudioSegment
import utils.utility as utility
import pyaudio

from pyannote.audio import Pipeline

from colorama import init as colorama_init
from colorama import Fore, Style

# Shared queue to store file paths produced by the producer
shared_queue = queue.Queue()
transcription_queue = queue.Queue()

# Event to signal the producer to stop producing
stop_event = threading.Event()


def compute_diarization(pipeline, audio_chunk_path, audio_seg, max_speakers_number, speakers_samples, speaker_emb_model):
    """
    Compute speaker diarization on the given audio chunk and return the resulting segments.
    :param pipeline: pipeline used to compute speaker diarization
    :param audio_chunk_path: path of the audio chunk to process
    :param audio_seg: AudioSegment object of the audio chunk to process
    :param args: arguments passed to the program via command line
    :param speakers_samples: dictionary containing the pre-recorded embedding of each speaker
    :param speaker_emb_model: model used to compute the embedding of the speakers
    :return: audio segments for each speaker as list of tuples (speaker name, audio segment)
    """
    diarization = pipeline(audio_chunk_path, max_speakers=max_speakers_number)
    audio_segments = utility.fuze_segments_and_audiofiles(diarization, audio_seg, speakers_samples, speaker_emb_model)
    return audio_segments


if __name__ == "__main__":
    model_path = "../models"
    color_list = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.WHITE, Fore.CYAN]
    language = "it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "medium"
    colorama_init()

    # Load / Download models
    print(f"Loading whisper {model_size} model {language}...")
    audio_model = whisper.load_model(model_size, download_root=model_path,
                                     device=device)

    print("Loading pyannote pipeline...")
    config_path = model_path + "/config_2_dontuse.yaml"
    dev = torch.device(device)
    try:
        pipeline = Pipeline.from_pretrained(config_path).to(dev)
    except HFValidationError:
        print("The config file may be not valid. Please check segmentation path and try again.")
        exit(1)

    print("Loading speaker embedding model...")
    speaker_embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                         device=torch.device(device))

    speakers_embeddings = []
    speakers = {}
    entries = os.listdir("../speakers")
    max_speakers_number = len(entries)
    for i, entry in enumerate(entries):
        print(entry)
        filepath = "../speakers/"+entry
        audio_seg = AudioSegment.from_wav(filepath)
        speaker_name = entry.split(".")[0]
        speakers[speaker_name] = (color_list[i % len(color_list)])
        audio_embedding = utility.from_audiosegment_to_embedding(speaker_embedding_model, audio_seg)
        speakers_embeddings.append((speaker_name, audio_embedding))

    last_speaker = None
    full_transcription = []

    for chunk in os.listdir("../tmp"):
        audio_chunk_path = "../tmp/"+chunk
        chunk_name = chunk.split(".")[0]
        segment = AudioSegment.from_wav(audio_chunk_path)

        audio_segments = compute_diarization(pipeline, audio_chunk_path, segment, max_speakers_number, speakers_embeddings,
                                             speaker_embedding_model)

        for pos, elem in enumerate(audio_segments):
            speaker = elem[0]
            segment = elem[1]
            segment.export("../segments/"+chunk_name+"_segment"+str(pos)+".wav", format="wav")

            audio_tensor = utility.pydub_to_np(segment)
            start_time = time.time()
            result = audio_model.transcribe(audio=audio_tensor, language=language, fp16=torch.cuda.is_available())
            end_time = (time.time() - start_time)
            duration_seconds = len(segment) / 1000  # Convert milliseconds to seconds
            print("--- "+chunk_name+"_segment"+str(pos)+".wav transcription "+ str(end_time)+ " seconds - duration " + str(duration_seconds))
            text = result['text'].strip()
            if text == "" or "sottotitoli e revisione a cura di qtss" in text.lower():
                continue

            #if last_speaker != speaker:
            #    print(f"------------------ {speaker} ------------------")
            #    last_speaker = speaker

            color = speakers[speaker]
            full_transcription.append(color + text + Style.RESET_ALL)
            print(color + text + Style.RESET_ALL)

print("full transcription")
for elem in full_transcription:
    print(elem)