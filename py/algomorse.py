
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt

# This is a real-time algorithm
# We load BLOCKSIZE samples at once.

BLOCKSIZE = 512
FFTSIZE = 8192
SPECTRAL_MEDIAN_FILTER = 51
PROMINENCE_THRESH = 25 # in dB

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
    # spectral median filter size must be odd-length
    assert(SPECTRAL_MEDIAN_FILTER % 2 == 1)
    self.blocks_per_fft = FFTSIZE/BLOCKSIZE
    self.block_buf = []
    self.block_i = 0
    self.pwrs = []
    self.peakss = []

  def input_block_cb(self,block):
    # block is byte array of BLOCKSIZE samples
    self.block_i += 1
    self.block_buf.append(block)
    # do an FFT about twice per second
    if self.block_i % self.blocks_per_fft == 0:
      self.find_sigs()

  def find_sigs(self):
    # make windowed copy of input
    fft_a = np.concatenate(self.block_buf[self.block_i-self.blocks_per_fft:])
    assert(FFTSIZE == len(fft_a))
    fft_a *= np.hanning(FFTSIZE) 
    # real fft
    fft_b = np.fft.rfft(fft_a)
    # compute power spectrum (in dB)
    pwr = np.log(np.square(np.real(fft_b)))*10
    # filter noise from spectrum
    pwr = median_filter(pwr,SPECTRAL_MEDIAN_FILTER)
    self.pwrs.append(pwr)
    # find peaks
    peaks = prominence(pwr,PROMINENCE_THRESH)
    self.peakss.append(peaks)

def block_u16_to_float(block):
  return np.array(block)/(float(2**15))

def median_filter(a,n):
  # a is array of floats
  # n is window size (odd integer)
  rv = np.zeros(len(a))
  hn = (n-1)/2
  for i in range(len(a)):
    mini = max(0,i-hn)
    maxi = min(len(a)-1,i+hn)
    med = np.median(a[mini:maxi+1])
    rv[i] = med
  return rv

def prominence(pwr,thresh):
  peaks = []
  absmin = np.min(pwr)
  absmax = np.max(pwr)
  lmin = absmax
  rmin = absmax
  maxv = absmin
  maxi = 0
  LEFT = -1
  RIGHT = +1
  state = LEFT
  for i in range(len(pwr)):
    v = pwr[i]
    if state == LEFT:
      lmin = min(lmin,v)
      if v - lmin > thresh:
        state = RIGHT
        rmin = absmax
    if state == RIGHT:
      rmin = min(rmin,v)
      if v > maxv:
        maxv = v
        maxi = i
        rmin = v
      if maxv - rmin > thresh:
        peaks.append(maxi)
        lmin = absmax
        rmin = absmax
        maxv = absmin
        state = LEFT
  return peaks

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

  for i in range(len(am.pwrs)):
    fig,ax = plt.subplots()
    ax.plot(am.pwrs[i])
    for j in range(len(am.peakss[i])):
      ax.axvline(x=am.peakss[i][j])
    plt.show()

  """fig,ax = plt.subplots()
  ax.imshow(data)
  plt.show()"""

  print "EOF"

