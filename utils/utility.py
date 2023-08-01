# code example https://github.com/Majdoddin/nlp/blob/main/Pyannote_plays_and_Whisper_rhymes_v_2_0.ipynb

# need to accept terms and conditions to use speaker diarization and pyannote
# https://huggingface.co/pyannote/speaker-diarization
# https://huggingface.co/pyannote/segmentation


# check https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/overlapped_speech_detection.ipynb
import whisper

from pyannote.audio import Pipeline
import numpy as np
from pyannote.audio import Audio

import re
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

import glob
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pydub import AudioSegment

from scipy.spatial.distance import cdist
from pathlib import Path

colorama_init()


def prepare_audio(audio_path, export_wav=False):
    """
    Prepare audio for pyannote diarization by appending two seconds of silence at the beginning of the audio file
    :param audio_path: str path to and audio file
    :param export_wav: bool whether to export the audio as wav file in the same directory of the input audio file
    :return: AudioSegment object of the audio file
    """

    # append two seconds of silence at the beginning of the audio file

    spacer = AudioSegment.silent(duration=2000)
    audio = spacer + AudioSegment.from_file(audio_path)

    if export_wav:
        # given a path like ./audio/song.mp3 replace all type of extension with .wav
        wav_path = re.sub(r"\.[a-zA-Z0-9]+", ".wav", audio_path)
        audio.export(wav_path, format='wav')

    return audio


def millisec(time):
    return int(time * 1000)


def pyannote_diarization(audio_path, config_path="", num_speakers=2, device="cuda"):
    """
    Generate pyannote diarization result and save it in the same directory of the input audio file as txt file
    :param audio_path: path to the audio file
    :param config_path: path config.yaml file of pyannote segmentation model name
    :param use_auth_token: hugingface.co auth token
    :param cache_dir: path to the cache directory
    :param device: bool to use cuda or cpu
    :return: diarization result as list of strings
    """

    # load pyannote pipeline
    pipeline = Pipeline.from_pretrained(config_path).to(device)

    # generate pyannote diarization result
    diarization = pipeline(audio_path, num_speakers=num_speakers)

    return diarization


def from_audiosegment_to_embedding(model, audio_segment):
    """
    Convert numpy array to torch tensor
    :return:
    """
    audio_np = pydub_to_np(audio_segment)
    audio_tensor = torch.from_numpy(audio_np).float()
    audio_tensor = audio_tensor[None]
    embedding = model(audio_tensor[None])

    return embedding


def assign_speaker(model, speakers_samples, audio_segment):
    """
    Assign speaker to each audio segment
    :param model: Speaker Embeddings model
    :param speakers_samples: list of tuples (Speaker Name, Speaker Embedding)
    :return: Speaker Name of the audio segment with the minimum cosine distance
    """
    # Convert the audio segment to torch embedding
    embedding = from_audiosegment_to_embedding(model, audio_segment)

    # Calculate the distance between the embedding of the audio segment and the embedding of each speaker, return the speaker with the minimum distance
    min_distance = 100
    speaker_name = None
    for speaker in speakers_samples:
        distance = cdist(speaker[1], embedding, metric="cosine")
        if distance < min_distance:
            min_distance = distance
            speaker_name = speaker[0]

    return speaker_name


def fuze_segments_and_audiofiles(diarization, audio_segment, speakers_samples, speaker_emb_model):
    """
    Fuze the diarization segments and the audio file into a list of tuples
    :param diarization: pyannote diarization result
    :param audio_segment: Original chunk of audio from which the diarization was extracted
    :param speakers_samples: list of tuples (Speaker Name, Speaker Embedding)
    :param speaker_emb_model: Speaker Embeddings model
    :return: list of tuples (speaker name, audio segment)
    """
    # Variables to keep track of the current speaker segment
    current_speaker = None
    segment_start = None
    segment_end = None
    audio_segments = []

    # Iterate through the diarization segments
    for speech_turn, track, speaker in diarization.itertracks(yield_label=True):

        # Check if the speaker changes
        if speaker != current_speaker:
            # Process the previous speaker segment, if it exists
            if current_speaker is not None:
                # Extract the corresponding portion of audio
                speaker_audio = audio_segment[segment_start:segment_end]

                duration_seconds = len(speaker_audio) / 1000  # Convert milliseconds to seconds

                if duration_seconds < 0.7:
                    continue

                # Detect the speaker name
                speaker_name = assign_speaker(speaker_emb_model, speakers_samples, audio_segment)

                audio_segments.append([speaker_name, speaker_audio])

            # Update the current speaker and segment boundaries
            current_speaker = speaker
            segment_start = millisec(speech_turn.start)

        # Update the end of the segment
        segment_end = millisec(speech_turn.end)

    # Process the last speaker segment
    if current_speaker is not None:
        # Extract the corresponding portion of audio
        speaker_audio = audio_segment[segment_start:segment_end]

        duration_seconds = len(speaker_audio) / 1000  # Convert milliseconds to seconds

        if duration_seconds > 0.7:
            # Detect the speaker name
            speaker_name = assign_speaker(speaker_emb_model, speakers_samples, audio_segment)

            audio_segments.append([speaker_name, speaker_audio])

    return audio_segments


# to implement in order to avoid saving of wav files
def pydub_to_np(audio_segment):
    """
    Convert pydub audio segment to numpy array
    :param audio_segment: AudioSegment object
    :return: numpy array of the audio segment
    """
    if audio_segment.frame_rate != 16000:  # 16 kHz
        audio_segment = audio_segment.set_frame_rate(16000)
    if audio_segment.sample_width != 2:  # int16
        audio_segment = audio_segment.set_sample_width(2)
    if audio_segment.channels != 1:  # mono
        audio_segment = audio_segment.set_channels(1)
    arr = np.array(audio_segment.get_array_of_samples())
    arr = arr.astype(np.float32) / 32768.0

    return arr


def transcribe_segments(audio_segments, segments, model="medium", device="cpu",
                        download_root="../models", language="en"):
    audio_model = whisper.load_model(model, download_root=download_root,
                                     device=device)
    colorama_init()
    speakers = {
        'SPEAKER_00': (Fore.RED),
        'SPEAKER_01': (Fore.GREEN)
    }

    # transcribe each segment
    for i in range(segments):
        file_pattern = f"{i}_*.wav"
        matching_files = glob.glob(file_pattern)[0]

        speaker = matching_files.split("_", 1)[1].split(".")[0]

        result = audio_model.transcribe(audio=matching_files, language=language, word_timestamps=True,
                                        fp16=torch.cuda.is_available())  # , initial_prompt=result.get('text', ""))
        text = result['text'].strip()
        color = speakers[speaker]
        print(color + text + Style.RESET_ALL)


def test_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path = "../audio_test_files/noisy_conversation.mp3"
    audio_seg = prepare_audio(audio_path)
    diarization = pyannote_diarization(audio_path, config_path="../models/config.yaml", num_speakers=2, device=device)
    audio_segments, segments = fuze_segments_and_audiofiles(diarization, audio_segment=audio_seg)
    transcribe_segments(audio_segments, segments, device=device)

# test_pipeline()
