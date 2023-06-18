# code example https://github.com/Majdoddin/nlp/blob/main/Pyannote_plays_and_Whisper_rhymes_v_2_0.ipynb

# need to accept terms and conditions to use speaker diarization and pyannote
# https://huggingface.co/pyannote/speaker-diarization
# https://huggingface.co/pyannote/segmentation

import whisper
import torch

from pydub import AudioSegment
from pyannote.audio import Pipeline

import re
from termcolor import colored


# append two seconds of silence at the beginning of the audio file
spacer = AudioSegment.silent(duration=2000)
audio = spacer + AudioSegment.from_mp3("../audio_test_files/two_speakers_test_4minutes.mp3")
audio.export('test_audio.wav', format='wav')

# pyannote diarization
model_name = "pyannote/speaker-diarization@2.1"
use_auth_token = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token, cache_dir="../models").to(device)

# generate pyannote diarization result
DEMO_FILE = {'uri': 'blabla', 'audio': 'test_audio.wav'}
dz = pipeline(DEMO_FILE)

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))


def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s



dzs = open('diarization.txt').read().splitlines()

groups = []
g = []
lastend = 0

for d in dzs:
    if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
        groups.append(g)
        g = []

    g.append(d)

    end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
    end = millisec(end)
    if (lastend > end):  # segment engulfed by a previous segment
        groups.append(g)
        g = []
    else:
        lastend = end
if g:
    groups.append(g)


# save audio segments for each group
audio = AudioSegment.from_wav("test_audio.wav")
gidx = -1
for g in groups:
  start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
  end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
  start = millisec(start) #- spacermilli
  end = millisec(end)  #- spacermilli
  gidx += 1
  audio[start:end].export(str(gidx) + '.wav', format='wav')


# load model
import whisper
audio_model = whisper.load_model('medium', download_root="../models", device="cuda" if torch.cuda.is_available() else "cpu")

# transcribe each segment

import json
for i in range(len(groups)):
  audiof = str(i) + '.wav'
  result = audio_model.transcribe(audio=audiof, language='en', word_timestamps=True, fp16=torch.cuda.is_available())#, initial_prompt=result.get('text', ""))
  with open(str(i)+'.json', "w") as outfile:
    json.dump(result, outfile, indent=4)


speakers = {
    'SPEAKER_00':('green'),
    'SPEAKER_01':('red')
}

from datetime import timedelta

def timeStr(t):
  return '{0:02d}:{1:02d}:{2:06.2f}'.format(round(t // 3600),
                                                round(t % 3600 // 60),
                                                t % 60)

spacermilli = 2000
txt = list("")
gidx = -1
for g in groups:
    shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
    shift = millisec(shift) - spacermilli  # the start time in the original video
    shift = max(shift, 0)

    gidx += 1

    captions = json.load(open(str(gidx) + '.json'))['segments']

    if captions:
        speaker = g[0].split()[-1]

        if speaker in speakers:
            color = speakers[speaker][0]

        for c in captions:
            start = shift + c['start'] * 1000.0
            start = start / 1000.0  # time resolution ot youtube is Second.
            end = (shift + c['end'] * 1000.0) / 1000.0
            txt.append(f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')

            print(colored(c["text"], color))

#with open(f"fullcaption.txt", "w", encoding='utf-8') as file:
#  s = "".join(txt)
#  file.write(s)
#  print('captions saved to fullcaption.txt:')
#  print(s+'\n')