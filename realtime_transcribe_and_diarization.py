import argparse

import whisper
import torch
import threading

from huggingface_hub.utils import HFValidationError
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import queue
import numpy as np
from datetime import datetime

from time import sleep
from sys import platform

from colorama import init as colorama_init
from colorama import Fore, Style

from pyannote.audio import Pipeline
from pydub import AudioSegment
import pyaudio

import utils.utility as utility

# Shared queue to store file paths produced by the producer
shared_queue = queue.Queue()

# Event to signal the producer to stop producing
stop_event = threading.Event()

def record_audio(filename, duration):
    chunk = 1024  # Number of frames per buffer

    # Set up the audio stream
    audio_format = pyaudio.paInt16  # 16-bit resolution
    channels = 1  # Mono audio
    sample_rate = 16000  # Sampling rate (Hz)

    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("The recording will start in 10 seconds... you need to read a sentence\n\n")
    print("In un luogo di mare c'era una volta un pesce di nome Nemo.\n"
          "Nemo era diverso dagli altri pesci perché aveva una pinna più piccola.\n"
          "A causa di questa sua differenza, spesso si sentiva escluso e triste.\n"
          "Ma un giorno, mentre nuotava alla ricerca di avventure, Nemo incontrò una simpatica piovra di nome Olivia.\n"
          "Olivia era molto curiosa e aveva tante storie interessanti da raccontare.\n"
          "Diventarono subito grandi amici e Nemo capì che la sua diversità non lo rendeva meno speciale.\n"
          "Insieme, Nemo e Olivia continuarono il loro viaggio attraverso l'oceano, scoprendo meraviglie e vivendo avventure indimenticabili.\n")

    sleep(5)
    print("Recording...")

    frames = []

    # Record audio for the specified duration
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording.")
    # Convert the recorded audio frames to AudioSegment
    audio_data = b"".join(frames)
    audio = AudioSegment(
        data=audio_data,
        sample_width=2,
        channels=channels,
        frame_rate=sample_rate
    )

    audio.export("./speakers/" + filename + ".wav", format="wav")
    return audio


def compute_diarization(pipeline, audio_chunk_path, audio_seg, args, speakers_samples, speaker_emb_model):
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
    diarization = pipeline(audio_chunk_path, max_speakers=args.max_speakers_number)
    audio_segments = utility.fuze_segments_and_audiofiles(diarization, audio_seg, speakers_samples, speaker_emb_model)
    return audio_segments


def producer(args, pipeline, speakers_samples, speaker_emb_model):
    """
    Producer thread used to record chunks of audio, compute speaker diarization and put the results in the shared queue.
    :param args: arguments passed to the program via command line
    :param pipeline: pipeline used to compute speaker diarization
    :param speakers_samples: dictionary containing the prerecorded embedding of each speaker
    :param speaker_emb_model: model used to compute the embedding of the speakers
    """

    # Define the parameters for recording
    sample_rate = 16000  # Sampling rate in Hz
    duration = args.record_timeout  # Duration of each audio segment in seconds
    chunk_size = int(sample_rate * duration)  # Number of samples in each chunk

    def callback(in_data, frame_count, time_info, status):
        # save timestamp in a variable called k
        k = datetime.now().strftime("%Y%m%d-%H%M%S")

        audio_data = np.frombuffer(in_data, dtype=np.int16)
        # Create an AudioSegment object from the audio data
        segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 2 bytes per sample for 16-bit audio
            channels=1,  # Mono audio
        )
        audio_chunk_path = f"./tmp/{k}_audio_chunk.wav"
        segment.export(audio_chunk_path, format='wav')

        audio_segments = compute_diarization(pipeline, audio_chunk_path, segment, args, speakers_samples,
                                             speaker_emb_model)

        for elem in audio_segments:
            shared_queue.put(elem)

        return None, pyaudio.paContinue

    # Create an instance of the PyAudio class
    pa = pyaudio.PyAudio()

    # Open a stream for recording audio
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
        stream_callback=callback
    )

    # Start the stream
    stream.start_stream()

    input("Recording audio. Press Enter to stop...")

    stop_event.set()
    print("Stopping recording...")

    # Stop the stream
    stream.stop_stream()

    # Close the stream and terminate PyAudio
    stream.close()
    pa.terminate()


def consumer(audio_model, language, speakers):
    """
    Consumer thread that takes audio segments from the shared queue and transcribes them.
    :param audio_model: whisper model
    :param language: language to use for the transcription
    :param speakers: dictionary containing the colorama color for each speaker {speaker: color}
    """
    colorama_init()
    full_transcription = []
    while True:
        try:
            [speaker, audio_seg] = shared_queue.get(timeout=1)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        audio_tensor = utility.pydub_to_np(audio_seg)
        result = audio_model.transcribe(audio=audio_tensor, language=language, word_timestamps=True,
                                        fp16=torch.cuda.is_available())
        text = result['text'].strip()
        if text == "":
            continue

        color = speakers[speaker]

        full_transcription.append(speaker + " " + text)
        print(f"------------------ {speaker} ------------------")
        print(color + text + Style.RESET_ALL)

        # Infinite loops are bad for processors, must sleep.
        shared_queue.task_done()
        sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--max_speakers_number", default=2, help="Maximum number of speakers to detect", type=int)

    parser.add_argument("--model_path", default="./models", help="Path where to download/load the model")

    parser.add_argument("--language", default="en", help="Language to use for the model")

    parser.add_argument("--record_timeout", default=7,
                        help="How real time the recording is in seconds.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    model = args.model
    model_path = args.model_path
    speakers = {}
    duration = args.record_timeout
    color_list = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.WHITE, Fore.CYAN]
    language = args.language
    if language == "en":
        model += ".en"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_speakers_number = args.max_speakers_number
    speakers_embeddings = []

    # Load / Download models
    print(f"Loading whisper {model} model {language}...")
    audio_model = whisper.load_model(model, download_root=model_path,
                                     device=device)

    print("Loading pyannote pipeline...")
    config_path = args.model_path + "/config.yaml"
    #dev = torch.device(device)
    try:
        pipeline = Pipeline.from_pretrained(config_path).to(device)
    except HFValidationError:
        print("The config file may be not valid. Please check segmentation path and try again.")
        exit(1)

    print("Loading speaker embedding model...")
    speaker_embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                         device=torch.device(device))

    for i in range(max_speakers_number):
        # ask the user to input the name of the speaker
        speaker_name = input("Insert the name of the speaker: ")

        # add a new speaker to the dictionary with a color
        speakers[speaker_name] = (color_list[i % len(color_list)])

        # record an audio sample for the speaker and save the audio file
        audio_seg = record_audio(speaker_name, 10)
        audio_embedding = utility.from_audiosegment_to_embedding(speaker_embedding_model, audio_seg)
        speakers_embeddings.append((speaker_name, audio_embedding))

    # Create and start the producer and consumer threads
    producer_thread = threading.Thread(target=producer,
                                       args=(args, pipeline, speakers_embeddings, speaker_embedding_model))
    consumer_thread = threading.Thread(target=consumer, args=(audio_model, language, speakers))

    producer_thread.start()
    consumer_thread.start()

    shared_queue.join()

    # Wait for the threads to complete
    producer_thread.join()
    consumer_thread.join()

if __name__ == "__main__":
    main()
