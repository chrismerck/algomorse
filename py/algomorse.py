
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# This is a real-time algorithm
# We load BLOCKSIZE samples at once.

BLOCKSIZE = 512
FFTSIZE = 8192
SPECTRAL_MEDIAN_FILTER = 51
PROMINENCE_THRESH = 25 # in dB
CW_FILTER = 200 # filter width
ALLOWED_DRIFT = 100
DECODER_TIMEOUT_S = 2

# Amount of Smoothing to Apply before Element Detection
# do not set higher than dit length
#  where dit length in ms is T = 1200 / WPM 
SMOOTHING_MS = 10 
DECODER_AGC_TIME_MS = 300

# We work in floats, initially scaled [-1.0,+1.0],
#  to avoid fixed-point complications.
# The embedded algorithm will need to be fixed-point.
#
# block = [ -.02, -.01, +.03, +.23, +.20, ... ]
#

def fits2hz(fits):
  # input in "fits" i.e. index in power spectrum
  # output in Hz
  return int(round(fits*samprate/float(FFTSIZE)))

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def exp_avg(a,alpha):
  rv = np.zeros(len(a))
  x = 0
  for i in range(len(a)):
    x = (1-alpha) * x + alpha*a[i]
    rv[i] = x
  return rv

class Decoder(object):
  def __init__(self,freq):
    self.freq = freq
    self.prev_block = np.zeros(BLOCKSIZE)
    self.ys = []
    self.ys2 = []
    self.age = 0

  def input_block(self,block):
    self.age += BLOCKSIZE/float(samprate)
    # stuff previous block behind this one
    #  to simulate a continuous filter
    x = np.concatenate([self.prev_block,block])
    # tight butterworth filter around signal
    lowcut = self.freq - CW_FILTER/2
    highcut = self.freq + CW_FILTER/2
    y = butter_bandpass_filter(x, lowcut, highcut, FFTSIZE, order=3)
    # rectify and smooth signal
    alpha = 1-np.exp(-2*np.pi*60/float(samprate))
    y = exp_avg(np.abs(y),alpha)
    # decimate signal to 1/10 sample-rate
    y = y[::10]
    y2 = []
    self.ys.append(y[len(y)/2:])
    self.ys2.append(y2[len(y2)/2:])
    self.prev_block = block

  def update_freq(self,freq):
    self.age = 0
    self.freq = freq

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
    self.decoders = []
    self.old_decoders = []

  def input_block(self,block):
    # block is byte array of BLOCKSIZE samples
    self.block_i += 1
    self.block_buf.append(block)
    # do an FFT about twice per second
    if self.block_i % self.blocks_per_fft == 0:
      self.find_sigs()
    for (freq,decoder) in self.decoders:
      decoder.input_block(block)

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
    # find matching decoders, creating if need-be
    for peak in peaks:
      match = False
      for (freq,decoder) in self.decoders:
        if abs(peak-freq) < ALLOWED_DRIFT:
          # found matching decoder
          # update decoder frequency
          decoder.update_freq(peak)
          match = True
          break
      # create new decoder
      if not match:
        print "New decoder at %d Hz"%fits2hz(peak)
        decoder = Decoder(peak)
        self.decoders.append((peak,decoder))
      surviving_decoders = []
      for (freq,decoder) in self.decoders:
        if decoder.age < DECODER_TIMEOUT_S:
          surviving_decoders.append((freq,decoder))
        else:
          # for now we keep the old decoders
          self.old_decoders.append((freq,decoder))
          print "Decoder timeout %d Hz"%fits2hz(freq)
      self.decoders = surviving_decoders

    # load 

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
  # find peaks in pwr array
  # with prominence of at least thresh 
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

  block_i = 0
  while True:
    block = inwave.readframes(BLOCKSIZE)
    if len(block) != BLOCKSIZE*sampwidth:
      break
    block = struct.unpack("<%dh"%BLOCKSIZE,block)
    block = block_u16_to_float(block)
    am.input_block(block)
    block_i += 1
    if block_i > 30:
      break

  decoders = am.old_decoders + am.decoders
  decoders.sort() # sort by increasing frequency

  fig,axs = plt.subplots(len(decoders),sharex=True)
  for i in range(len(decoders)):
    freq,decoder = decoders[i]
    axs[i].plot(np.concatenate(decoder.ys))
    axs[i].plot(np.concatenate(decoder.ys2))
  plt.show()

  """fig,ax = plt.subplots()
  ax.imshow(data)
  plt.show()"""

  print "EOF"

