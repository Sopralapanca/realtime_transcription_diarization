# tutorial on tds https://towardsdatascience.com/unlock-the-power-of-audio-data-advanced-transcription-and-diarization-with-whisper-whisperx-and-ed9424307281
# github page of the tutorial https://github.com/luisroque/large_laguage_models/blob/main/speech2text_whisperai_pyannotate.py
# package to download https://github.com/m-bain/whisperX
# voice dataset https://media.talkbank.org/ca/CallHome/eng/

# https://colab.research.google.com/drive/1HuvcY4tkTHPDzcwyVH77LCh_m8tP-Qet?usp=sharing provare
# https://github.com/Majdoddin/nlp/blob/main/Pyannote_plays_and_Whisper_rhymes_v_2_0.ipynb

# important discussion to follow https://github.com/openai/whisper/discussions/264

# need to accept terms and conditions to use speaker diarization and pyannote
# https://huggingface.co/pyannote/speaker-diarization
# https://huggingface.co/pyannote/segmentation

import whisper
import torch
from typing import Optional, List, Dict, Any
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

def transcribe(audio_file: str, model_name: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Transcribe an audio file using a speech-to-text model.

    Args:
        audio_file: Path to the audio file to transcribe.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the transcript, including the segments, the language code, and the duration of the audio file.
    """
    model = whisper.load_model(model_name, download_root="../models", device=device)
    result = model.transcribe(audio_file, language="en", fp16=torch.cuda.is_available())

    language_code = result["language"]
    return {
        "segments": result["segments"],
        "language_code": language_code,
    }


def align_segments(
        segments: List[Dict[str, Any]],
        language_code: str,
        audio_file: str,
        device: str = "cpu"):
    """
    Align the transcript segments using a pretrained alignment model.

    Args:
        segments: List of transcript segments to align.
        language_code: Language code of the audio file.
        audio_file: Path to the audio file containing the audio data.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the aligned transcript segments.
    """
    model_a, metadata = load_align_model(language_code=language_code, device=device, model_dir="../models")
    result_aligned = align(segments, model_a, metadata, audio_file, device)
    return result_aligned


def diarize(audio_file: str, hf_token: str, device: str) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_file: Path to the audio file to diarize.
        hf_token: Authentication token for accessing the Hugging Face API.

    Returns:
        A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
    """
    diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarization_result = diarization_pipeline(audio_file)
    return diarization_result


def assign_speakers(
        diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker diarization result.

    Args:
        diarization_result: Dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    result_segments, word_seg = assign_word_speakers(
        diarization_result, aligned_segments["segments"]
    )
    results_segments_w_speakers: List[Dict[str, Any]] = []
    for result_segment in result_segments:
        results_segments_w_speakers.append(
            {
                "start": result_segment["start"],
                "end": result_segment["end"],
                "text": result_segment["text"],
                "speaker": result_segment["speaker"],
            }
        )
    return results_segments_w_speakers


def transcribe_and_diarize(
        audio_file: str,
        hf_token: str,
        model_name: str,
        device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization to determine which words were spoken by each speaker.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    print("Transcribing audio file...")
    transcript = transcribe(audio_file, model_name, device)
    print(transcript)
    print("Aligning transcript segments...")
    aligned_segments = align_segments(
        transcript["segments"], transcript["language_code"], audio_file, device
    )
    print(aligned_segments)
    print("Diarizing audio file...")
    diarization_result = diarize(audio_file, hf_token, device)
    print(diarization_result)
    print("Assigning speakers to transcript segments...")
    results_segments_w_speakers = assign_speakers(diarization_result, aligned_segments)
    print(results_segments_w_speakers)

    # Print the results in a user-friendly way
    for i, segment in enumerate(results_segments_w_speakers):
        print(f"Segment {i + 1}:")
        print(f"Start time: {segment['start']:.2f}")
        print(f"End time: {segment['end']:.2f}")
        print(f"Speaker: {segment['speaker']}")
        print(f"Transcript: {segment['text']}")
        print("")

    return results_segments_w_speakers


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

audio_file = "../audio_test_files/two_speakers_test_4minutes.mp3"
hf_token = "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"
transcribe_and_diarize(audio_file, hf_token, "medium", device)
