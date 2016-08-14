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
  block_queue.put(np.transpose(indata)[0])
  print "Qlen = %d"%(block_queue.qsize())
  # playback
  outdata[:] = indata

with sd.Stream(samplerate=8000, channels=1, blocksize=512, callback=callback):
  while True:
    sd.sleep(100)

