import whisper
import torch

model = whisper.load_model("small", download_root="../models", device="cuda" if torch.cuda.is_available() else "cpu")
result = model.transcribe("../audio_test_files/two_speakers_test_4minutes.mp3")
print(result["text"])