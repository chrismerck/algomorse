import sounddevice as sd
import algomorse
import numpy as np
from threading import Thread
from Queue import Queue, Empty
import pygame

def text_cb(d):
  text_queue.put(d)

samprate = 8000
am = algomorse.Algomorse(samprate,text_cb)

block_queue = Queue()
text_queue = Queue()
shutdown_queue = Queue()

def task():
  while True:
    block = block_queue.get(block=True)
    am.input_block(block)

t=Thread(target=task)
t.daemon=True
t.start()

def pygame_task():
  pygame.init()
  font = pygame.font.SysFont('Calibri', 25, True, False)
  size = (1000,500)
  screen = pygame.display.set_mode(size)
  done = False
  clock = pygame.time.Clock()
  decoder_text = dict()
  while not done:
    for event in pygame.event.get(): # User did something
      if event.type == pygame.QUIT: # If user clicked close
        done = True # Flag that we are done so we exit this loop
    try:
      d = text_queue.get_nowait()
      if d['evt'] == 'new_decoder':
        decoder_text[d['id']] = {'hz':d['hz'],'text':''}
      elif d['evt'] == 'decoder_timeout':
        del decoder_text[d['id']]
      elif d['evt'] == 'decoder_output':
        decoder_text[d['id']]['text'] += d['text']
        if len(decoder_text[d['id']]['text']) > 50:
          decoder_text[d['id']]['text'] = decoder_text[d['id']]['text'][1:]
      elif d['evt'] == 'update_hz':
        decoder_text[d['id']]['hz'] = d['hz']
      else:
        print "unknown event: '%s'"%d['evt']
    except Empty:
      pass
    # --- Drawing code should go here
    screen.fill((0,0,0))
    strings = []
    for did in decoder_text.keys():
      d = decoder_text[did]
      strings.append((d['hz'],'[%d] %s'%(d['hz'],d['text'])))
    strings.sort()
    for i in range(len(strings)):
      text = font.render(strings[i][1],True,(0,255,0))
      screen.blit(text, (10,i*30))
    pygame.display.flip()
    clock.tick(60) # limit to 60fps
  print "[Got Quit Event]"
  shutdown_queue.put([])

t2=Thread(target=pygame_task)
t2.daemon=True
t2.start()

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
  while shutdown_queue.qsize() == 0:
    sd.sleep(100)
  print "[Stream Shutdown]"


