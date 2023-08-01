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

# Convert raw audio bytes to an AudioSegment object.
def bytes_to_audio_segment(raw_data, sample_width=2, frame_rate=16000, channels=1):
    audio_segment = AudioSegment.from_file(io.BytesIO(raw_data), sample_width=sample_width,
                                           frame_rate=frame_rate, channels=channels)
    return audio_segment

def recorder(max_speakers, pipeline, speakers_samples, speaker_emb_model):
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1500
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    #recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    record_timeout = 10
    phrase_timeout = 2

    transcription = ['']
    if source is not None:
        with source:
            recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    i=0
    print("Start recording. Press CTRL+C to interrupt.")
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                # phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    # phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                k = datetime.now().strftime("%Y%m%d-%H%M%S")
                audio_chunk_path = f"./tmp/{k}_audio_chunk.wav"

                segment = AudioSegment.from_file(io.BytesIO(last_sample),
                                                 format='raw',
                                                 sample_width=source.SAMPLE_WIDTH,
                                                 frame_rate=source.SAMPLE_RATE, channels=1)

                segment.export(audio_chunk_path, format="wav")

                audio_segments = compute_diarization(pipeline, audio_chunk_path, segment, max_speakers, speakers_samples,
                                                     speaker_emb_model)

                for elem in audio_segments:
                    elem.append(i)
                    print("producer putting",i)
                    shared_queue.put(elem)
                    i += 1

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


def consumer(audio_model, language):
    """
    Consumer thread that takes audio segments from the shared queue and transcribes them.
    :param audio_model: whisper model
    :param language: language to use for the transcription
    :param speakers: dictionary containing the colorama color for each speaker {speaker: color}
    """

    while True:
        try:
            [speaker, audio_seg, position] = shared_queue.get(timeout=1)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        print("consumer working on ", position)
        audio_tensor = utility.pydub_to_np(audio_seg)
        result = audio_model.transcribe(audio=audio_tensor, language=language, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        if text == "":
            continue

        transcription_queue.put([speaker, text, position])

        # Infinite loops are bad for processors, must sleep.
        shared_queue.task_done()
        sleep(0.25)

def printer(speakers):
    colorama_init()
    last_speaker = None
    transcription = []
    full_transcription = []
    while True:
        try:
            [speaker, text, position] = transcription_queue.get(timeout=1)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        print("printing", position)
        color = speakers[speaker]
        full_transcription.append(speaker + " " + text)

        if last_speaker != speaker:
            print(f"------------------ {speaker} ------------------")
            last_speaker = speaker

        print(color + text + Style.RESET_ALL)
        transcription_queue.task_done()
        sleep(0.25)

        # If we detected a pause between recordings, add a new item to our transcripion.
        # Otherwise edit the existing one.
        # if phrase_complete:
        #    transcription.append(text)
        # else:
        #    transcription[-1] = text

        # Clear the console to reprint the updated transcription.
        # os.system('cls' if os.name == 'nt' else 'clear')
        # for line in transcription:
        #    print(line)
        # Flush stdout.
        # print('', end='', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--max_speakers_number", default=2, help="Maximum number of speakers to detect", type=int)

    parser.add_argument("--model_path", default="./models", help="Path where to download/load the model")

    parser.add_argument("--language", default="en", help="Language to use for the model")

    parser.add_argument("--record_timeout", default=7,
                        help="How real time the recording is in seconds.", type=float)

    parser.add_argument("--use_saved_speakers", action="store_true",
                        help="wav files in speakers directory will be used as speakers samples. The name of the "
                             "speakers will be the name of the saved file.")
    args = parser.parse_args()

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

    speakers_embeddings = []

    # Load / Download models
    print(f"Loading whisper {model} model {language}...")
    audio_model = whisper.load_model(model, download_root=model_path,
                                     device=device)

    print("Loading pyannote pipeline...")
    config_path = args.model_path + "/config.yaml"
    dev = torch.device(device)
    try:
        pipeline = Pipeline.from_pretrained(config_path).to(dev)
    except HFValidationError:
        print("The config file may be not valid. Please check segmentation path and try again.")
        exit(1)

    print("Loading speaker embedding model...")
    speaker_embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                         device=torch.device(device))

    saved_speakers = args.use_saved_speakers
    if saved_speakers:
        entries = os.listdir("./speakers")
        max_speakers_number = len(entries)
        for i, entry in enumerate(entries):
            filepath = "./speakers/"+entry
            audio_seg = AudioSegment.from_wav(filepath)
            speaker_name = entry.split(".")[0]
            speakers[speaker_name] = (color_list[i % len(color_list)])
            audio_embedding = utility.from_audiosegment_to_embedding(speaker_embedding_model, audio_seg)
            speakers_embeddings.append((speaker_name, audio_embedding))
    else:
        max_speakers_number = args.max_speakers_number
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
    consumer_thread = threading.Thread(target=consumer, args=(audio_model, language))
    printer_thread = threading.Thread(target=printer, args=(speakers, ))

    consumer_thread.start()
    printer_thread.start()


    recorder(max_speakers_number, pipeline, speakers_embeddings, speaker_embedding_model)

    shared_queue.join()
    transcription_queue.join()

    # Wait for the threads to complete
    consumer_thread.join()
    printer_thread.join()

