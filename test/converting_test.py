from pydub import AudioSegment
import numpy as np
import whisper
import torch

language = "it"
model = "medium"
model_path = "../models"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading whisper {model} model {language}...")
audio_model = whisper.load_model(model, download_root=model_path, device=device)

# load wav file with pydub
audio_path = "20230611-004139_audio_chunk.wav"
audio_segment = AudioSegment.from_wav(audio_path)
sample_rate = audio_segment.frame_rate

#audio_segment = audio_segment.low_pass_filter(1000)
# convert to expected format
if audio_segment.frame_rate != 16000: # 16 kHz
    audio_segment = audio_segment.set_frame_rate(16000)
if audio_segment.sample_width != 2:   # int16
    audio_segment = audio_segment.set_sample_width(2)
if audio_segment.channels != 1:       # mono
    audio_segment = audio_segment.set_channels(1)
arr = np.array(audio_segment.get_array_of_samples())
arr = arr.astype(np.float32)/32768.0

print(f"Transcribing...")
result = audio_model.transcribe(arr, language=language, fp16=torch.cuda.is_available())
text = result['text'].strip()
print(text)

waveform = arr
audio_segment = AudioSegment(
    waveform.tobytes(),
    frame_rate=sample_rate,
    sample_width=waveform.dtype.itemsize,
    channels=1
)
audio_segment.export("test.wav", format="wav")



print(f"Transcribing...")
result = audio_model.transcribe(audio_path, language=language, fp16=torch.cuda.is_available())
text = result['text'].strip()
print(text)