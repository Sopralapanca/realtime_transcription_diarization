import socket
import torch
import os
import io
import json
import struct
from pydub import AudioSegment
import whisper
import time
import sys
from pyannote.audio import Pipeline
from huggingface_hub.utils import HFValidationError
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import pickle

sys.path.append("./utils")
import utility

# Server configuration
HOST = '0.0.0.0'  # Listen on all available interfaces (public IP)
PORT = 8080
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(DEVICE)
MODEL_PATH = "./models"
MODEL_SIZE = "medium"
LANG = "it"

speakers_embeddings = []


def reconstruct_audio_segment(audio_data, frame_rate=44100, sample_width=2, channels=1):
    audio_stream = io.BytesIO(audio_data)
    audio_segment = AudioSegment.from_raw(audio_stream, frame_rate=frame_rate, sample_width=sample_width,
                                          channels=channels)  # Adjust parameters according to your AudioSegment configuration
    return audio_segment


def receive_data(conn):
    # Read the header to get the chunk size
    header = conn.recv(4)
    if not header:
        return None
    chunk_size = struct.unpack('>I', header)[0]

    # Read the chunk data using the chunk size
    data = b''
    remaining_bytes = chunk_size

    while remaining_bytes > 0:
        chunk = conn.recv(min(4096, remaining_bytes))
        if not chunk:
            break
        data += chunk
        remaining_bytes -= len(chunk)

    if not data:
        return None

    return data


def handle_store_speaker(speaker_name, data, speaker_embedding_model):
    audio_segment = reconstruct_audio_segment(data)
    audio_segment.export(f"./server_speakers/{speaker_name}.wav", format="wav")

    audio_embedding = utility.from_audiosegment_to_embedding(speaker_embedding_model, audio_segment)
    speakers_embeddings.append((speaker_name, audio_embedding))

    print("Stored speaker: " + speaker_name)


def handle_DAT(audio_model, LANG, pipeline, speaker_embedding_model, data, max_speakers_number, speakers_embeddings):
    audio_segment = reconstruct_audio_segment(data)
    # get time to store file with name+timestamp
    timestamp = time.time()
    audio_chunk_path = f"./server_chunks/chunk-{timestamp}.wav"
    audio_segment.export(audio_chunk_path, format="wav")

    audio_segments = utility.compute_diarization(pipeline, audio_chunk_path, audio_segment, max_speakers_number,
                                                 speakers_embeddings,
                                                 speaker_embedding_model)

    for elem in audio_segments:
        speaker_name = elem[0]
        segment = elem[1]

        audio_tensor = utility.pydub_to_np(segment)
        start_time = time.time()
        result = audio_model.transcribe(audio=audio_tensor, language=LANG, fp16=torch.cuda.is_available())
        end_time = (time.time() - start_time)
        duration_seconds = len(audio_segment) / 1000  # Convert milliseconds to seconds
        text = result['text'].strip()

        response_dict = {
            "response": "ok",
            "segment_duration": duration_seconds,
            "transcription_time": end_time,
            "speaker_name": speaker_name,
            "text": text
        }

        return response_dict

def handle_client_connection(conn, audio_model, LANG, pipeline, speaker_embedding_model):
    while True:
        print("Waiting for data...")
        data = receive_data(conn)
        if not data:
            break

        data = pickle.loads(data)
        message_type = data["type"]
        message_data = data["data"]

        if message_type == "STORE_SPEAKER":
            speaker_name = data["speaker_name"]
            print("storing speaker: " + speaker_name)
            handle_store_speaker(speaker_name, message_data, speaker_embedding_model)
            response_message = {
                "status": "SUCCESS",
                "message": "Audio segment stored successfully."
            }

            # Serialize and send the response back to the client
            response_data = pickle.dumps(response_message)
            response_header = struct.pack('>I', len(response_data))
            conn.sendall(response_header + response_data)
        # if message_type == "LOAD_SPEAKERS":
        #    speaker_names = message_data["speaker_name"]
        #    r = handle_store_speaker(speaker_name, data, speaker_embedding_model)

        if message_data == "DAT":
            max_speakers_number = len(speakers_embeddings)
            print("max speakers number: " + str(max_speakers_number))
            response = handle_DAT(audio_model, LANG, pipeline, speaker_embedding_model, data, max_speakers_number, speakers_embeddings)

            response = json.dumps(response).encode()
            conn.send(response + b'\n')

def main():
    print("Loading pyannote pipeline...")
    config_path = "./models/config.yaml"

    try:
        pipeline = Pipeline.from_pretrained(config_path)
    except HFValidationError:
        print(f"The config file in {config_path} may be not valid. Please check segmentation path and try again.")
        exit(1)

    print("Loading speaker embedding model...")
    speaker_embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                         device=dev)

    # Load / Download models
    print(f"Loading whisper {MODEL_SIZE} model {LANG}...")
    audio_model = whisper.load_model(MODEL_SIZE, download_root=MODEL_PATH,
                                     device=DEVICE)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Waiting for connections...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    handle_client_connection(conn, audio_model, LANG, pipeline, speaker_embedding_model)

    conn.close()


if __name__ == "__main__":
    main()
