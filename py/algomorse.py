
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
PROMINENCE_THRESH = 20 # in dB
CW_FILTER = 200 # filter width
ALLOWED_DRIFT = 100
DECODER_TIMEOUT_S = 5

# Amount of Smoothing to Apply before Element Detection
RECTIFY_HZ = 40
# adjustment for fading
THRESHOLD_AVG_HZ = 0.8
#MIN_THRESH = 0.01 # unknown units
MIN_DIT_MS = 24
MIN_DAH_SCALE = 2
MAX_DAH_SCALE = 5

ELEM_STAT_N = 5 # minimum elements needed for statistics

# We work in floats, initially scaled [-1.0,+1.0],
#  to avoid fixed-point complications.
# The embedded algorithm will need to be fixed-point.
#
# block = [ -.02, -.01, +.03, +.23, +.20, ... ]
#

morse = {
  '.-':'a',
  '-...':'b',
  '-.-.':'c',
  '-..':'d',
  '.':'e',
  '..-.':'f',
  '--.':'g',
  '....':'h',
  '..':'i',
  '.---':'j',
  '-.-':'k',
  '.-..':'l',
  '--':'m',
  '-.':'n',
  '---':'o',
  '.--.':'p',
  '--.-':'q',
  '.-.':'r',
  '...':'s',
  '-':'t',
  '..-':'u',
  '...-':'v',
  '.--':'w',
  '-..-':'x',
  '-.--':'y',
  '--..':'z',
  '.----':'1',
  '..---':'2',
  '...--':'3',
  '....-':'4',
  '.....':'5',
  '-....':'6',
  '--...':'7',
  '---..':'8',
  '----.':'9',
  '-----':'0',
  '.-.-.-':'.',
  '--..--':',',
  '---...':':',
  '..--..':'?',
  '.----.':"'",
  '-..-.':'/',
  '.-.-.':'<AR>',
  '-.--.':'<KN>',
  '-...-.-':'<BK>',
  '.'*8:'<HH>',
  '...-.-':'<SK>',
  '...-.-':'<SK>',
  '...---...':'<SOS>',
}


def fits2hz(fits,samprate):
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

def exp_avg(a,alpha,x0=0):
  rv = np.zeros(len(a))
  x = x0
  for i in range(len(a)):
    x = (1-alpha) * x + alpha*a[i]
    rv[i] = x
  return rv

KEY_UP = 0
KEY_DOWN = 1
class Decoder(object):
  def __init__(self,freq,samprate,text_cb=lambda x: x,flags=[]):
    self.text_cb = text_cb
    self.flags = flags
    self.samprate = samprate
    self.freq = freq
    self.prev_block = np.zeros(BLOCKSIZE)
    self.ys = []
    self.ys2 = []
    self.age = 0
    self.timer = 0
    self.thresh = 0
    self.hist = 0.6 # as fraction of thresh
    self.key_evts = []
    self.key_down_time = 0
    self.key_up_time = 0
    self.state = KEY_UP
    self.next_evt = 0 # next event to be decoded
    self.Tdit = 60.0 # ms
    self.dah_scale = 2
    self.Tdit_alpha = 0.5
    self.elems_log = ""
    self.elems = ""
    self.text = ""
    self.last_t_up = 0
    self.last_evt_thresh = 0

  def input_block(self,block):
    # stuff previous block behind this one
    #  to simulate a continuous filter
    x = np.concatenate([self.prev_block,block])
    # tight butterworth filter around signal
    lowcut = self.freq - CW_FILTER/2
    highcut = self.freq + CW_FILTER/2
    y = butter_bandpass_filter(x, lowcut, highcut, FFTSIZE, order=3)
    y_sav = y[len(y)/2:][::10]
    # rectify and smooth signal
    alpha = 1-np.exp(-2*np.pi*RECTIFY_HZ/float(self.samprate))
    y = exp_avg(np.abs(y),alpha)
    maxy = np.max(y)
    # decimate signal to 1/10 sample-rate
    y = y[::10]
    # remove unused first half (prev block)
    y = y[len(y)/2:]
    decimated_samprate = self.samprate/10.0
    alpha = 1-np.exp(-2*np.pi*THRESHOLD_AVG_HZ/float(decimated_samprate))
    # detect elements
    for i in range(len(y)):
      # threshold is slow exponential average
      self.thresh = (1-alpha) * self.thresh + alpha * y[i]
      #self.thresh = max(MIN_THRESH, self.thresh)
      if self.state == KEY_UP:
        if y[i] > self.thresh:
          self.key_down_time = self.age + i/float(decimated_samprate)
          self.state = KEY_DOWN
      elif self.state == KEY_DOWN:
        if y[i] < self.thresh * self.hist:
          self.key_up_time = self.age + i/float(decimated_samprate)
          self.state = KEY_UP
          if (self.key_up_time - self.key_down_time)*1000.0 < MIN_DIT_MS:
            # dit too short
            pass
          else:
            self.key_evts.append((self.key_down_time,self.key_up_time,self.thresh))
            self.decode_machine()
            #print "[%d] Key Event: (%f,%f)"%(self.freq,self.key_down_time,self.key_up_time)
    self.age += BLOCKSIZE/float(self.samprate)
    self.timer += BLOCKSIZE/float(self.samprate)
    if 'ys' in self.flags:
      self.ys.append(y)
    self.prev_block = block

  def update_freq(self,freq):
    self.timer = 0
    self.freq = freq
    hz = fits2hz(freq,self.samprate)
    self.text_cb({'evt':'update_hz','id':id(self),'hz':hz})

  def decode_machine(self):
    decimated_samprate = self.samprate/10.0
    for i in range(self.next_evt,len(self.key_evts)):
      e = self.key_evts[i]
      this_evt_thresh = e[2]
      # force word separation if a much stronger signal comes in
      if this_evt_thresh > 3*self.last_evt_thresh:
        self.elems = ''
        self.text += '|'
      self.last_evt_thresh = this_evt_thresh
      dur_ms = (e[1] - e[0])*1000.0
      gap_ms = (e[0] - self.last_t_up)*1000.0
      self.last_t_up = e[1]
      # determine element/letter/word spacing
      eol = False
      eow = False
      if gap_ms < (self.Tdit * (1 + self.dah_scale))/2.0:
        # element space
        pass
      elif gap_ms < (self.Tdit * (5.5 + 1.5*self.dah_scale))/2.0:
        # letter space
        eol = True
      else:
        # word space
        eol = True
        eow = True
      if eol:
        if not self.elems in morse:
          # unknown letter (probably noise)
          #print "ERROR: unknown letter: ",self.elems
          self.text += '_'
          self.elems_log += '%s '%self.elems
          self.text_cb({'evt':'decoder_output','id':id(self),'text':'_'})
        else:
          self.text += morse[self.elems]
          self.elems_log += '(%s) '%self.elems
          self.text_cb({'evt':'decoder_output','id':id(self),'text':morse[self.elems]})
        self.elems = ''
      if eow:
        self.text += ' '
        self.elems_log += '|'
        self.text_cb({'evt':'decoder_output','id':id(self),'text':' '})
        print "[%d] %s"%(fits2hz(self.freq,self.samprate),self.text)

      # determine dit or dah
      if dur_ms < self.Tdit * (1+self.dah_scale)/2.0:
        # recognized a dit
        #print "DIT dur_ms=%f Tdit=%f"%(dur_ms,self.Tdit)
        self.Tdit = (1-self.Tdit_alpha)*self.Tdit + self.Tdit_alpha*dur_ms
        self.elems += '.'
      else:
        # recognized a dah
        #print "DAH dur_ms=%f Tdit=%f"%(dur_ms,self.Tdit)
        #self.dah_scale = (1-self.Tdit_alpha)*self.dah_scale + self.Tdit_alpha*(dur_ms/self.Tdit)
        self.Tdit = (1-self.Tdit_alpha)*self.Tdit + self.Tdit_alpha*(dur_ms/self.dah_scale)
        self.elems += '-'
    self.next_evt = len(self.key_evts)


class Algomorse(object):
  def __init__(self,samprate,text_cb=lambda x: x,flags=[]):
    self.flags = flags
    self.samprate = samprate
    # fft size must be integer number of blocks
    assert(FFTSIZE % BLOCKSIZE == 0)
    # spectral median filter size must be odd-length
    assert(SPECTRAL_MEDIAN_FILTER % 2 == 1)
    self.blocks_per_fft = FFTSIZE/BLOCKSIZE
    self.block_buf = []
    self.block_i = 0
    self.pwrs = []
    self.peakss = []
    self.decoders = []
    self.old_decoders = []
    self.text_cb = text_cb

  def input_block(self,block):
    # block is byte array of BLOCKSIZE samples
    self.block_i += 1
    self.block_buf.append(block)
    assert(len(block) == BLOCKSIZE)
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
    # release blocks once they are two FFT periods old
    self.block_buf = self.block_buf[-self.blocks_per_fft:]
    self.block_i = len(self.block_buf)
    # real fft
    fft_b = np.fft.rfft(fft_a)
    # compute power spectrum (in dB)
    pwr = np.log(np.square(np.real(fft_b)))*10
    # filter noise from spectrum
    pwr = median_filter(pwr,SPECTRAL_MEDIAN_FILTER)
    ## remove filtershape from spectrum
    #pwr_avg = exp_avg(pwr,0.005,pwr[0])
    #pwr = pwr - pwr_avg
    if 'pwrs' in self.flags:
      self.pwrs.append(pwr)
    # find peaks
    peaks = prominence(pwr,PROMINENCE_THRESH)

    # filter out false peaks
    peaks2 = []
    for peak in peaks:
      if peak > 100 and peak < len(pwr)-100 \
          and pwr[peak] - pwr[peak-100] > 15 and pwr[peak] - pwr[peak+100] > 15:
        peaks2.append(peak)
    peaks = peaks2

    if 'peakss' in self.flags:
      self.peakss.append(peaks)
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
        hz = fits2hz(peak,self.samprate)
        print "New decoder at %d Hz"%hz
        decoder = Decoder(peak,self.samprate,text_cb=self.text_cb,flags=self.flags)
        self.text_cb({'evt':'new_decoder','id':id(decoder),'hz':hz})
        # shove the previous blocks into the decoder,
        #  so it can see the onset of the signal
        for block in self.block_buf:
          decoder.input_block(block)
        self.decoders.append((peak,decoder))
      surviving_decoders = []
      for (freq,decoder) in self.decoders:
        if decoder.timer < DECODER_TIMEOUT_S:
          surviving_decoders.append((freq,decoder))
        else:
          # for now we keep the old decoders
          if 'old_decoders' in self.flags:
            self.old_decoders.append((freq,decoder))
          self.text_cb({'evt':'decoder_timeout','id':id(decoder)})
          print "Decoder timeout %d Hz"%fits2hz(freq,self.samprate)
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
  infilename = "../cub40m3.wav"

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

  am = Algomorse(samprate,flags=['peakss','pwrs','old_decoders','ys'])

  block_i = 0
  while True:
    block = inwave.readframes(BLOCKSIZE)
    if len(block) != BLOCKSIZE*sampwidth:
      break
    block = struct.unpack("<%dh"%BLOCKSIZE,block)
    block = block_u16_to_float(block)
    am.input_block(block)
    block_i += 1
    if block_i > 100:
      pass #break

  """
  for i in range(len(am.pwrs)):
    fig,ax = plt.subplots()
    ax.plot(am.pwrs[i])
    for j in range(len(am.peakss[i])):
      ax.axvline(x=am.peakss[i][j])
    plt.show() 
  """
  
  decoders = am.old_decoders + am.decoders
  decoders.sort() # sort by increasing frequency

  fig,axs = plt.subplots(len(decoders),sharex=True)
  if len(decoders) == 1:
    axs = [axs]
  for i in range(len(decoders)):
    freq,decoder = decoders[i]
    print "[%d] %s"%(decoder.freq,decoder.text)
    print "[%d] %s"%(decoder.freq,decoder.elems_log)
    y = np.concatenate(decoder.ys)
    axs[i].plot(np.arange(len(y))/(samprate/10.0), y)
    avgy = np.mean(y)
    for j in range(len(decoder.key_evts)):
      key_evt = decoder.key_evts[j]
      axs[i].plot([key_evt[0],key_evt[1]],[avgy/1.5,avgy*1.5],'k-',lw=3)
      axs[i].set_ylabel("[%d]"%decoder.freq)
  plt.show()

  """fig,ax = plt.subplots()
  ax.imshow(data)
  plt.show()"""

  print "EOF"

