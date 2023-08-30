import socket
import json
from huggingface_hub.utils import HFValidationError
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pydub import AudioSegment
import sys
sys.path.append("../utils")
import utility
import pyaudio
import torch
from pyannote.audio import Pipeline
import os
import struct
from colorama import init as colorama_init
from colorama import Fore, Style
import time

# Server configuration
HOST = '172.17.0.1'
PORT = 12345
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Adjust as needed
COLOR_LIST = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.WHITE, Fore.CYAN]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(DEVICE)


def compute_diarization(pipeline, audio_chunk_path, audio_seg, max_speakers_number, speakers_samples,
                        speaker_emb_model):
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


def receive_all_data(conn, chunk_size=1024):
    data = b""
    while True:
        try:
            chunk = conn.recv(chunk_size)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:  # Use a delimiter to separate responses
                break
        except EOFError:
            print("EOFError")
            break

    return data


def establish_connection(remote_server_ip, port):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((remote_server_ip, port))
            print("Connection to the server established.")
            return client_socket
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            print("Retrying in 5 seconds...")
            time.sleep(5)


def main():
    colorama_init()

    print("Loading pyannote pipeline...")
    config_path = "../models/config_2_dontuse.yaml"

    try:
        pipeline = Pipeline.from_pretrained(config_path)
    except HFValidationError:
        print(f"The config file in {config_path} may be not valid. Please check segmentation path and try again.")
        exit(1)

    print("Loading speaker embedding model...")
    speaker_embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                         device=dev)
    print("Starting client...")
    client_socket = establish_connection(HOST, PORT)

    speakers_embeddings = []
    speakers = {}
    entries = os.listdir("../speakers")
    max_speakers_number = len(entries)
    for i, entry in enumerate(entries):
        filepath = "../speakers/" + entry
        audio_seg = AudioSegment.from_wav(filepath)
        speaker_name = entry.split(".")[0]
        speakers[speaker_name] = (COLOR_LIST[i % len(COLOR_LIST)])
        audio_embedding = utility.from_audiosegment_to_embedding(speaker_embedding_model, audio_seg)
        speakers_embeddings.append((speaker_name, audio_embedding))

    last_speaker = None
    full_transcription = []

    for chunk in os.listdir("../tmp"):
        audio_chunk_path = "../tmp/" + chunk
        chunk_name = chunk.split(".")[0]
        segment = AudioSegment.from_wav(audio_chunk_path)

        audio_segments = compute_diarization(pipeline, audio_chunk_path, segment, max_speakers_number,
                                             speakers_embeddings,
                                             speaker_embedding_model)

        for pos, elem in enumerate(audio_segments):
            speaker = elem[0]
            segment = elem[1]
            segment_path = "../segments/" + chunk_name + "_segment" + str(pos) + ".wav"

            # segment.export(segment_path, format="wav")

            # if last_speaker != speaker:
            #    print(f"------------------ {speaker} ------------------")
            #    last_speaker = speaker

            color = speakers[speaker]

            audio_data = segment.raw_data
            audio_size = len(audio_data)

            header = struct.pack('I', audio_size)  # Pack the chunk size as a 4-byte integer
            client_socket.sendall(header + audio_data)

            # The client should now receive the server's response
            response = receive_all_data(client_socket, chunk_size=1024)

            response_dict = json.loads(response.decode())
            print(f"Text: {response_dict['text']} --- duration {response_dict['segment_duration']} transcription {response_dict['transcription_time']}")

            # full_transcription.append(color + text + Style.RESET_ALL)
            # print(color + text + Style.RESET_ALL)

    client_socket.close()


if __name__ == "__main__":
    main()
