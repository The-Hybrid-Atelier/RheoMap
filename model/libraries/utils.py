# import librosa
import os, time
from pathlib import Path
import numpy as np

__name__ = "utils"

def delete_x_lines(x):
  for i in range(x):
    print ("\033[A                             \033[A")
def si_unit(value, unit):
  prefix = ""
  if value < 1:
    prefix = "m"
    value = value*1000.0
  elif value > 999:
    prefix = "k"
    value = value/1000
  elif value > 999999:
    prefix = "M"
    value = value/1000000
  return "%2.1f %s%s"%(value, prefix, unit)
  
def ohm(value):
  return si_unit(value, "Ω")
def volts(value):
  return si_unit(value, "V")
def interp(a, min=-1, max=1):
    return np.interp(a, (a.min(), a.max()), (min, max))


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def header(message, l=0):
  if l > 0:
    prefix = " "*l
  else:
    prefix = ""
  suffix = ""
  if l == -2:
    prefix = prefix + bcolors.OKGREEN
    suffix = bcolors.ENDC
  if l == -1:
    prefix = prefix + bcolors.WARNING
    suffix = bcolors.ENDC
  if l == 0:
    prefix = prefix + bcolors.BOLD
    suffix = bcolors.ENDC
  elif l == 1:
    prefix = prefix + bcolors.OKBLUE
    suffix = bcolors.ENDC
  elif l == 2:
    prefix = prefix + bcolors.UNDERLINE
    suffix = bcolors.ENDC
  print(prefix, message, suffix)

def key_copy(obj, keys):
  cp = {}
  for k in keys:
    cp[k] = obj[k]
  return cp

def pretty_print_dictionary(d, prefix="  ", callback=None):
  for key in d:
    try:
      if type(d[key]) == dict:
        print(prefix + key)
        pretty_print_dictionary(d[key], prefix + "\t", callback)
      else:
        if callback:
         print("%s%s %2.2f "%(prefix, key, callback(d[key])))
        else:
          print("%s%s %2.2f "%(prefix, key, d[key]))
    except TypeError:
      print("%s%s %s"%(prefix, key, d[key]))

  
def f1(precision, recall):
  return 2*(recall * precision) / (recall + precision)

# def read_file(filename, path='', sample_rate=None, trim=False):
#     ''' Reads in a flac file and returns it as an np.float32 array in the range [-1,1] '''
#     filename = Path(path) / filename
#     data, file_sr = librosa.load(filename)
#     if data.dtype == np.int16:
#         data = np.float32(data) / np.iinfo(np.int16).max
#     elif data.dtype != np.float32:
#         raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))
#     if sample_rate is not None and sample_rate != file_sr:
#         if len(data) > 0:
#             data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
#         file_sr = sample_rate
#     if trim and len(data) > 1:
#         data = librosa.effects.trim(data, top_db=40)[0]
#     return data, file_sr

def file_timestamp(prefix="", extension=""):
  return prefix+str(time.time()).replace(".", "_")+"."+extension

def make_dir(path):
  if not os.path.exists(path):
    print("Creating %s folder"%(os.path.relpath(path)))
    os.mkdir(path)
    return path
  return path

def custom_timestamp():
  return int(round(time.time() * 1000))


def millis_to_vtt(count, end, start, label):
  
  seconds = (start/1000)%60
  minutes = (start/(60*1000))%60
  hours = (start/(60*60*1000))%24

  elapsed_seconds = ((end)/1000)%60
  elapsed_minutes = ((end)/(60*1000))%60
  elapsed_hours = ((end)/(60*60*1000))%24
  count += 1

  return "\n%i\n%02i:%02i:%06.3f  -->  %02i:%02i:%06.03f \n%s\n"%(count, hours, minutes, seconds, 
                                                  elapsed_hours, elapsed_minutes, elapsed_seconds, label)
