
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt

# This is a real-time algorithm
# We load BLOCKSIZE samples at once.

BLOCKSIZE = 512
FFTSIZE = 8192

# We work in floats, initially scaled [-1.0,+1.0],
#  to avoid fixed-point complications.
# The embedded algorithm will need to be fixed-point.
#
# block = [ -.02, -.01, +.03, +.23, +.20, ... ]
#

class Algomorse(object):
  def __init__(self):
    # fft size must be integer number of blocks
    assert(FFTSIZE % BLOCKSIZE == 0)
    self.blocks_per_fft = FFTSIZE/BLOCKSIZE
    self.block_buf = []
    self.block_i = 0
    self.fft_bs = []

  def input_block_cb(self,block):
    # block is byte array of BLOCKSIZE samples
    self.block_i += 1
    self.block_buf.append(block)
    # do an FFT about twice per second
    if self.block_i % self.blocks_per_fft == 0:
      # make windowed copy of input
      fft_a = np.concatenate(self.block_buf[self.block_i-self.blocks_per_fft:])
      assert(FFTSIZE == len(fft_a))
      fft_a *= np.hanning(FFTSIZE) 
      fft_b = np.fft.rfft(fft_a)
      self.fft_bs.append(fft_b)


def block_u16_to_float(block):
  return np.array(block)/(float(2**15))

if __name__ == "__main__":
  infilename = "../websdr_recording_start_2016-08-14T11-24-45Z_14034.7kHz.wav"

  inwave = wave.open(infilename,'r')
  nframes = inwave.getnframes()
  samprate = inwave.getframerate()
  nchan = inwave.getnchannels()
  sampwidth = inwave.getsampwidth()

  print "File...........", infilename
  print "Frames.........", nframes
  print "Sample Rate....", samprate
  print "Channels.......", nchan
  print "Sample Width...", sampwidth

  # single-channel 16-bit
  assert(nchan == 1)
  assert(sampwidth == 2)

  am = Algomorse()

  while True:
    block = inwave.readframes(BLOCKSIZE)
    if len(block) != BLOCKSIZE*sampwidth:
      break
    block = struct.unpack("<%dh"%BLOCKSIZE,block)
    block = block_u16_to_float(block)
    am.input_block_cb(block)

  data = np.log(np.square(np.real(np.array(am.fft_bs))))
  print data

  fig,ax = plt.subplots()
  for i in range(len(data)):
    ax.plot(data[i])
  plt.show()

  """fig,ax = plt.subplots()
  ax.imshow(data)
  plt.show()"""

  print "EOF"

