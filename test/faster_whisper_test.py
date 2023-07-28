# https://github.com/guillaumekln/faster-whisper
# https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb
# Nvidia NeMo MSDD
import whisper
import torch
import time
from faster_whisper import WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#model = whisper.load_model("small", download_root="../models", device=DEVICE)
# take transcription time and print it
#start_time = time.time()
#result = model.transcribe("../audio_test_files/1001_IEO_DIS_HI.mp3", fp16=False) #fp16=False if runned on CPU
#end_time = time.time()
#print(result["text"])
#elapsed_time = end_time - start_time
#print("Elapsed time: " + str(elapsed_time))

model_size = "large-v2"
model = WhisperModel(model_size, device=DEVICE, compute_type="float32", download_root="../models")
segments, info = model.transcribe("../audio_test_files/1001_IEO_DIS_HI.mp3", beam_size=5)

print(info)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))