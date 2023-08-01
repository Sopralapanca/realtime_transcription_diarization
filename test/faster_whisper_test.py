# https://github.com/guillaumekln/faster-whisper
# https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb
# Nvidia NeMo MSDD
import whisper
import torch
import time
from faster_whisper import WhisperModel
import wave
import time
from pydub import AudioSegment
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device used", DEVICE)

def simulate_realtime_recording(mp3_file, sample_rate=16000, chunk_size=2048, model=None, faster=False):

    audio_segment = AudioSegment.from_mp3(mp3_file)

    # Split the AudioSegment into chunks based on the given chunk size
    chunks = audio_segment[::chunk_size]

    for chunk in chunks:

        if chunk.frame_rate != 16000:  # 16 kHz
            chunk = chunk.set_frame_rate(16000)
        if chunk.sample_width != 2:  # int16
            chunk = chunk.set_sample_width(2)
        if chunk.channels != 1:  # mono
            chunk = chunk.set_channels(1)

        arr = np.array(chunk.get_array_of_samples())
        audio_tensor = arr.astype(np.float32) / 32768.0

        if faster == False:
            start_time = time.time()
            result = model.transcribe(audio_tensor, fp16=True, language="it")  # fp16=False if runned on CPU
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Whisper: {result['text']} - Elapsed time: {elapsed_time}")
        else:
            start_time = time.time()
            segments, info = model.transcribe(audio_tensor, beam_size=5, best_of=5, language="it")
            segments = list(segments)  # the transcription will be done when iterating on the segments
            end_time = time.time()
            elapsed_time = end_time - start_time
            text = ""
            for s in segments:
                text += s.text + " "
            print(f"Faster: {text} - Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    mp3_file_path = "../audio_test_files/video_3minutes_italian.mp3"

    model = whisper.load_model("medium", download_root="../models", device=DEVICE)
    simulate_realtime_recording(mp3_file_path, model=model)

    #model_size = "large-v2"
    model = WhisperModel("medium", device=DEVICE, compute_type="float32", download_root="../models")
    simulate_realtime_recording(mp3_file_path, model=model, faster=True)
