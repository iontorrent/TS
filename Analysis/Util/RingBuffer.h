/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <iostream>
#include <pthread.h>

template<typename T>
class RingBuffer {

  T *_buffer;
  int _numBuffers;
  int _bufSize;
  int _readPos;
  int _writePos;

  pthread_mutex_t _lock;
  pthread_cond_t _rdcond;
  pthread_cond_t _wrcond;
  
public: 
  RingBuffer(int numBuf, int bufSize);
  ~RingBuffer();

  int getBufSize() const { return _bufSize; }
  int getNumBufers() const { return _numBuffers; }
  int getReadPos() const { return _readPos; }
  int getWritePos() const { return _writePos; }

  void updateReadPos();
  void updateWritePos();
  T* readOneBuffer();
  T* writeOneBuffer();
  
private:
  void AllocateBuffer();

};


#endif // RINGBUFFER_H
