import socket
import json
from pydub import AudioSegment
import sys

import pyaudio
import os
import struct
from colorama import init as colorama_init
from colorama import Fore, Style
import time
import pickle
from threading import Thread

# Server configuration
HOST = 'dbalboni.cc'

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Adjust as needed
COLOR_LIST = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.WHITE, Fore.CYAN]


def speaker_segment_sender(client_socket, speaker_name, audio_seg):
    """
    Send the given audio segment to the server and return the response.
    :param speaker_name: name of the speaker
    :param audio_seg: AudioSegment object of the audio chunk to send
    :return: response from the server
    """
    message = {
        "type": "STORE_SPEAKER",
        "speaker_name": speaker_name,
        "data": audio_seg.raw_data
    }
    data = pickle.dumps(message)

    # send data including the header with the dimension of the object
    client_socket.sendall(struct.pack('>I', len(data)) + data)

    # Receive response
    response = receive_all_data(client_socket)
    response = pickle.loads(response)
    print("response: ", response)
    return response


def receive_all_data(conn):
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


def receive_message(client_socket, speakers):
    response = receive_all_data(client_socket)
    response_dict = json.loads(response.decode())
    color = speakers[response_dict["speaker_name"]]
    text = response_dict["text"]
    print(f"--- duration {response_dict['segment_duration']} transcription {response_dict['transcription_time']}")
    print(color + text + Style.RESET_ALL)


def main():
    colorama_init()

    print("Starting client...")
    port = 13284  # get port from bore
    client_socket = establish_connection(HOST, port)

    speakers = {}
    entries = os.listdir("./speakers")


    for i, entry in enumerate(entries):
        filepath = "./speakers/" + entry
        audio_seg = AudioSegment.from_wav(filepath)
        speaker_name = entry.split(".")[0]
        speakers[speaker_name] = (COLOR_LIST[i % len(COLOR_LIST)])
        print("sending speaker " + speaker_name)
        sender = speaker_segment_sender(client_socket, speaker_name, audio_seg)

    # start a thread that waits for messages from the server
    receive_message_thread = Thread(target=receive_message, args=(client_socket, speakers))
    receive_message_thread.start()

    for entry in os.listdir("./tmp"):
        filepath = "./tmp/" + entry
        audio_seg = AudioSegment.from_wav(filepath)

        audio_data = audio_seg.raw_data
        audio_size = len(audio_data)

        header = struct.pack('I', audio_size)  # Pack the chunk size as a 4-byte integer
        print("sending audio chunk", entry)
        client_socket.sendall(header + audio_data)

    client_socket.close()


if __name__ == "__main__":
    main()
