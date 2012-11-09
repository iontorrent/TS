# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Sequence and read streaming utilities
import sys
import time

import pysam


# decorators section
def _use_and_del(d, k, v):
  """
  Use a values from a dict and remove it.
  """
  if k in d:
    v = d[k]
    del d[k]
  return v

def open_read_stream(fn):
  """
  Take a string or file argument and pass it to the function as an open file.
  """
  def wrapped(f, *args, **kargs):
    if type(f) is file:   return fn(f, *args, **kargs)
    elif type(f) is str:  return fn(open(f, 'r', 16384), *args, **kargs)
    else: raise Exception(str(f) + ' is not a string or file')
  return wrapped

def value_stream(fn):
  """
  Take an iterator that iterates over lines (e.g., from a file) and
  return the lines stripped and split into values using delimter.
  Skip blank lines and lines that start with characters in the
  skip_lines string.
  """
  
  def wrapped(f, *args, **kargs):
    
    skip_header = _use_and_del(kargs, 'skip_header', True)
    skip_blank = _use_and_del(kargs, 'skip_blank', True)
    skip_lines = _use_and_del(kargs, 'skip_lines', '#')
    delimiter = _use_and_del(kargs, 'delimiter', None)
    
    lines = fn(f, *args, **kargs)
    
    if skip_header:
      header = lines.next()
    
    for line in lines:
      values = line.strip().split(delimiter)
      if skip_blank and len(values) == 0: continue
      if values[0][0] in skip_lines: continue
      yield values
      
  return wrapped

def metered_stream(fn):
  """
  Display a progress meter to standard out for a stream.
  """

  def wrapped(*args, **kargs):

    meter = _use_and_del(kargs, 'meter', True)
    freq = _use_and_del(kargs, 'meter_freq', 10000)

    stream = fn(*args, **kargs)

    # Hijack the stream and keep count of the items
    if meter:
      count = 0
      start = time.time()

    for record in stream:
      yield record

      if meter:
        count += 1
        if count % freq == 0:
          tick = time.time()
          sys.stderr.write('\rProcessed %d lines %.2f lines/sec' % (
            count, count / (tick - start)))
          sys.stderr.flush()

    if meter:
      tick = time.time()
      sys.stderr.write('\rProcessed %d lines %.2f lines/sec\n' % (
        count, count / (tick - start)))
      sys.stderr.flush()

  return wrapped

# end of decorators section

@open_read_stream
@value_stream
def file_values(f, *args, **kargs):  return f

@metered_stream
def sam_stream(sam, meter=True, skip_unmapped=False, skip_reverse=False):
  """
  Use pysam instead...
  """
  stream = pysam.Samfile(sam)

  for read in stream:
    if skip_unmapped and read.is_unmapped:    continue # skip umapped reads
    if skip_reverse  and read.is_reverse: continue # skip reverse reads    
    if read.rname != -1:
      rname = stream.getrname(read.tid) # make rname visible to the caller
    else:
      rname = ''
    yield read, rname
    
  return
