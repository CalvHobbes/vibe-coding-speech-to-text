import sounddevice as sd
import numpy as np

print("Recording for 2 seconds...")
audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1)
sd.wait()
print("Done.") 