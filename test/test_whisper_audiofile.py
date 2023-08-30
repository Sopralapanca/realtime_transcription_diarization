import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = whisper.load_model("small", download_root="../models", device=device)
result = model.transcribe("../audio_test_files/1001_IEO_DIS_HI.mp3")
print(result["text"])