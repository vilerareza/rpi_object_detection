'''
Program to test the basic audio function using Pygame.
The speaker should be connected to 3.5 mm stereo jack on the RPi
'''

import time
from pygame import mixer

audio_path = 'audio_test.mp3'
vol = 0.7

# Initialize pygame mixer
mixer.init()
# Loading the audio
mixer.music.load(audio_path)
# Setting the volume
mixer.music.set_volume(vol)
# Play the audio file

while True:

    mixer.music.play()
    query = input("Press 'r' to play again or 'q' to quit")
    if query == 'q':
        break
    time.sleep(1)
