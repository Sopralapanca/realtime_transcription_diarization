import socket
import torch
import os
import io
import json
import struct
from pydub import AudioSegment
import whisper
import utils.utility as utility
import time

# Server configuration
HOST = '127.0.0.1'  # Listen on all available interfaces (public IP)
PORT = 12345


def reconstruct_audio_segment(audio_data):
    return AudioSegment.from_raw(
        audio_data,
        sample_width=2,  # 16-bit audio
        frame_rate=44100,
        channels=1  # Mono audio
    )

def receive_audio_data(conn, audio_model, language):
    audio_count = 0
    while True:
        # Read the header to get the chunk size
        header = conn.recv(4)
        if not header:
            break
        chunk_size = struct.unpack('I', header)[0]

        # Read the chunk data using the chunk size
        audio_data = b''
        while len(audio_data) < chunk_size:
            data = conn.recv(chunk_size - len(audio_data))
            if not data:
                break
            audio_data += data

        if not audio_data:
            break

        audio_stream = io.BytesIO(audio_data)
        audio_segment = AudioSegment.from_raw(audio_stream, frame_rate=16000, sample_width=2, channels=1)  # Adjust parameters according to your AudioSegment configuration

        audio_tensor = utility.pydub_to_np(audio_segment)
        start_time = time.time()
        result = audio_model.transcribe(audio=audio_tensor, language=language, fp16=torch.cuda.is_available())
        end_time = (time.time() - start_time)
        duration_seconds = len(audio_segment) / 1000  # Convert milliseconds to seconds
        text = result['text'].strip()

        response_dict = {
            "response": "ok",
            "segment_duration": duration_seconds,
            "transcription_time": end_time,
            "text": text
        }

        response = json.dumps(response_dict).encode()
        conn.send(response+ b'\n')

        audio_count += 1

def main():
    model_path = "../models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "medium"
    language = "it"

    # Load / Download models
    print(f"Loading whisper {model_size} model {language}...")
    audio_model = whisper.load_model(model_size, download_root=model_path,
                                     device=device)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Waiting for a connection...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    receive_audio_data(conn, audio_model, language)

    conn.close()


if __name__ == "__main__":
    main()
