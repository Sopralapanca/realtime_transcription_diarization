import torch
from pydub import AudioSegment
import utils.utility as utility
from scipy.spatial.distance import cdist
from pathlib import Path
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

from pyannote.audio import Audio
from pyannote.core import Segment
audio = Audio(sample_rate=16000, mono="downmix")

# extract embedding for a speaker speaking between t=3s and t=6s
#speaker1 = Segment(3., 6.)
#waveform1, sample_rate = audio.crop("audio.wav", speaker1)
#embedding1 = model(waveform1[None])

# extract embedding for a speaker speaking between t=7s and t=12s
#speaker2 = Segment(7., 12.)
#waveform2, sample_rate = audio.crop("audio.wav", speaker2)
#embedding2 = model(waveform2[None])

# load audiosegment with pydub from wav file
audio_segment_tinti = AudioSegment.from_wav("../audio_test_files/audio_tinti.wav")
audio_segment_zerocalcare = AudioSegment.from_wav("../audio_test_files/audio_zerocalcare2.wav")

audio_tinti_np = utility.pydub_to_np(audio_segment_tinti)
audio_zerocalcare_np = utility.pydub_to_np(audio_segment_zerocalcare)

# convert np to torch tensor with shape (channel, time)
audio_tinti_tensor = torch.from_numpy(audio_tinti_np).float()
audio_zerocalcare_tensor = torch.from_numpy(audio_zerocalcare_np).float()

# add channel dimension
audio_tinti_tensor = audio_tinti_tensor[None]
audio_zerocalcare_tensor = audio_zerocalcare_tensor[None]

embedding_tinti = model(audio_tinti_tensor[None])
embedding_zerocalcare = model(audio_zerocalcare_tensor[None])

# load each wav file in current directory
for wav_file in Path(".").glob("*.wav"):
    audio_segment = AudioSegment.from_wav(wav_file)
    audio_np = utility.pydub_to_np(audio_segment)
    audio_tensor = torch.from_numpy(audio_np).float()
    audio_tensor = audio_tensor[None]
    embedding = model(audio_tensor[None])

    distance = cdist(embedding_tinti, embedding, metric="cosine")
    print(f"Distance between Tinti and {wav_file}: {distance}")

    distance = cdist(embedding_zerocalcare, embedding, metric="cosine")
    print(f"Distance between Zerocalcare and {wav_file}: {distance}")
