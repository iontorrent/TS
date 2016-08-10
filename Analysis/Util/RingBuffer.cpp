/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RingBuffer.h"

template<typename T>
RingBuffer<T>::RingBuffer(int numBuf, int bufSize):
   _numBuffers(numBuf), _bufSize(bufSize)
{
  AllocateBuffer();
  _readPos = 0;
  _writePos = 0;

  pthread_mutex_init(&_lock, NULL);
  pthread_cond_init(&_rdcond, NULL);
  pthread_cond_init(&_wrcond, NULL);
}

template<typename T>
RingBuffer<T>::~RingBuffer()
{
  if (_buffer)
    delete _buffer;

  pthread_cond_destroy(&_rdcond);
  pthread_cond_destroy(&_wrcond);
  pthread_mutex_destroy(&_lock);
}

template<typename T>
void RingBuffer<T>::AllocateBuffer()
{
  _buffer = new T[_bufSize*_numBuffers];
}

template<typename T>
T* RingBuffer<T>::readOneBuffer()
{
  pthread_mutex_lock(&_lock);
 
  while (getReadPos() == getWritePos())
    pthread_cond_wait(&_rdcond, &_lock);

  T *buf = _buffer + getReadPos()*getBufSize();

  pthread_mutex_unlock(&_lock);

  return buf;
}

template<typename T>
T* RingBuffer<T>::writeOneBuffer()
{
  pthread_mutex_lock(&_lock);
 
  int futWritePos = (getWritePos() + 1) % getNumBufers();
  while (futWritePos == getReadPos())
    pthread_cond_wait(&_wrcond, &_lock);

  T *buf = _buffer + getWritePos()*getBufSize();

  pthread_mutex_unlock(&_lock);
  return buf;
}

template<typename T>
void RingBuffer<T>::updateReadPos() {
  pthread_mutex_lock(&_lock);

  _readPos = (getReadPos() + 1) % getNumBufers();

  pthread_cond_signal(&_wrcond);
  pthread_mutex_unlock(&_lock);
}

template<typename T>
void RingBuffer<T>::updateWritePos() {
  pthread_mutex_lock(&_lock);

  _writePos = (getWritePos() + 1) % getNumBufers();

  pthread_cond_signal(&_rdcond);
  pthread_mutex_unlock(&_lock);
}

template class RingBuffer<float>;

