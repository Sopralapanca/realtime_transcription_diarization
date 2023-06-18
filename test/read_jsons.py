import re
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()
speakers = {
    'SPEAKER_00':(Fore.RED),
    'SPEAKER_01':(Fore.GREEN)
}

# load file named fullcaption.txt
with open("fullcaption.txt", "r", encoding='utf-8') as file:
    txt = file.read()
    # a line of this file is made like this:

    # the speaker is either SPEAKER_00 or SPEAKER_01
    for line in txt.splitlines():
        # get the speaker from the line
        speaker = re.findall('\[SPEAKER_[0-9]+\]', string=line)[0].replace('[', '').replace(']', '').strip()
        # get the text from a string like this [00:00:00.000 --> 00:00:00.000] [SPEAKER_00] text
        text = re.findall('\[SPEAKER_[0-9]+\] (.*)', string=line)[0].strip()
        color = speakers[speaker]
        print(color + text + Style.RESET_ALL)






