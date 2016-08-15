import sounddevice as sd
import algomorse
import numpy as np
from threading import Thread
from Queue import Queue
duration = 5  # seconds

samprate = 8000
am = algomorse.Algomorse(samprate)

block_queue = Queue()

def task():
  while True:
    block = block_queue.get(block=True)
    am.input_block(block)

t=Thread(target=task)
t.daemon=True
t.start()

def callback(indata, outdata, frames, time, status):
  if status:
    print status
  block_queue.put(np.transpose(indata)[0]/float(2**15))
  qlen = block_queue.qsize()
  if qlen > 3:
    print "Qlen = %d"%qlen
  # playback
  outdata[:] = indata

with sd.Stream(samplerate=8000, channels=1, blocksize=512, callback=callback, dtype='int16'):
  while True:
    sd.sleep(100)

