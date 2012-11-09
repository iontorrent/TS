/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "cuda_runtime.h"

#include "dumper.h"


DumpBuffer::DumpBuffer(unsigned int bytes, const char * name)
{
  _buffer = new char[bytes];
  _sizeBytes=bytes;
  _writeOffset = 0;
  
  if(name != NULL) strcpy(_name, name);
  else _name[0] = '\0';
}

DumpBuffer::DumpBuffer(ifstream& myFile)
{
  _buffer = NULL;
  _sizeBytes=0;
  _writeOffset = 0;
  _name[0] = '\0';

  readFromFile(myFile);

}


DumpBuffer::~DumpBuffer()
{
  cleanup();
}

void DumpBuffer::cleanup()
{
  _name[0] = '\0';
  _writeOffset = 0;
  _sizeBytes = 0;
  if (_buffer != NULL) delete [] _buffer;
  _buffer = NULL;
}


unsigned int DumpBuffer::addData(void *data, unsigned int bytes)
{
  
  char * writePtr = _buffer + _writeOffset;
  _writeOffset += bytes;
  if(_writeOffset > _sizeBytes){
    cerr << "allocated buffer too small" << endl;
    exit (-1);
  }

  memcpy((void*)writePtr, data, bytes);

  return (_writeOffset - bytes);
}

unsigned int DumpBuffer::addCudaData(void *devData, unsigned int bytes)
{
  char * writePtr = _buffer + _writeOffset;
  _writeOffset += bytes;
  if(_writeOffset > _sizeBytes){
    cerr << "allocated buffer too small" << endl;
    exit (-1);
  }

  cudaMemcpy((void*)writePtr, devData, bytes, cudaMemcpyDeviceToHost );

  return (_writeOffset - bytes);
}


bool DumpBuffer::CompareCuda(float * devData, float threshold, OutputFormat output)
{
  
  DumpBuffer temp( getSize(), getName());
  temp.addCudaData(devData,getSize());
  return Compare(temp, threshold, output);
} 

void DumpBuffer::writeToFile(ofstream& myFile)
{

  myFile.write((const char*)&_writeOffset , sizeof(unsigned int));
  myFile.write((const char*)_name , sizeof(char)*DUMPNAMELEN);
  myFile.write((const char*)_buffer , _writeOffset);
}
 
void DumpBuffer::readFromFile(ifstream& myFile)
{

 
  cleanup();

  myFile.read((char*)&_sizeBytes,sizeof(unsigned int));
  myFile.read((char*)_name,sizeof(char)*DUMPNAMELEN);

 _buffer = new char[_sizeBytes];

  myFile.read((char*)_buffer, _sizeBytes);

}


void * DumpBuffer::getData()
{
  return (void *)_buffer; 
}


unsigned int DumpBuffer::getSize()
{
  return (_writeOffset > 0)?(_writeOffset):(_sizeBytes); 
}

const char * DumpBuffer::getName()
{
  return _name; 
}


bool DumpBuffer::CompareName(DumpBuffer& buffer)
{
  if (0 == strcmp(getName(), buffer.getName())) return true;
  return false;
}

bool DumpBuffer::CompareSize(DumpBuffer& buffer)
{
  if( getSize() == buffer.getSize()) return true;
  return false;
}

bool DumpBuffer::CompareData(DumpBuffer& buffer, float threshold, OutputFormat output, int length , int stride ,int start)
{
  unsigned int size = min(getSize(),buffer.getSize()); 
  size /= sizeof(float);

  return CompareData((float*)buffer.getData(), threshold, output, length ,stride ,start);
}

bool DumpBuffer::CompareData(float * data, float threshold, OutputFormat output, int length , int stride ,int start)
{
  unsigned int size = getSize()/sizeof(float); 
  float * ptrA = (float*)getData();
  float * ptrB = data;
  int n = 0;
  double stddev = 0;
  unsigned int cnt = 0;   
  int i = start;
  int l = 0;

  if(stride < 0 ) stride = 0;
 
  if(length > stride) size = ( length+start <= size)?(length+start):(size);
  
  while(i < size){


    float diff = abs(ptrA[i] - ptrB[i]);
    float error = abs(diff/((ptrA[i] != 0)?(ptrA[i]):(1)));
    if(error>threshold || error != error ){//&& diff > 0.01 ){ 
     if(output > MIN) cout << i <<" " <<  ptrA[i] << " " << ptrB[i] << " " <<  (100.0*error) << "%" << " F" << endl;  
     n++;
     if(error == error) stddev += diff*diff; 
    }else{
     if(output==ALL) cout << i << " " <<  ptrA[i] << " " << ptrB[i] << " " <<  (100.0*error) << "%" << endl;  
     stddev += diff*diff; 
    }

    cnt++; 
    l++;
   
    if(stride > length && length == l){
      l = 0;
      i += stride - length + 1; 
      if(output == ALL) cout << endl;   
    }else{
      i++;
    }
  }
  
  stddev =   sqrt(stddev/cnt); 
  
   cout << "compare " << getName()<<" threshold of " << threshold  << " failed for "<< n << " out of " << cnt << " ( " << (100.0 *n)/((double)(cnt))  << "% ), standard deviation: "<< stddev << endl;
  if(n==0) return true; 

  return false;
}



bool DumpBuffer::Compare(DumpBuffer& buffer, float threshold, OutputFormat output, int length , int stride ,int start)
{


  if(!CompareName(buffer)) return false;
  if(!CompareSize(buffer)){ 
    cout << "Buffer " << getName() << " different number of data points: " << getSize()/sizeof(float) << " / " << buffer.getSize()/sizeof(float) << endl;
    return false;
  }
  if(!CompareData(buffer, threshold, output, length , stride ,start)) return false;
  cout << _name << " correctness check PASSED" << endl;  
  return true;
}




bool DumpBuffer::CompareCuda(float * devData, float * hostData, unsigned int size, float threshold, OutputFormat output)
{

  DumpBuffer tmpCuda(size,"CudaBuffer");
  tmpCuda.addCudaData(devData,size);
  return tmpCuda.CompareData(hostData,threshold, output);

}



unsigned int DumpBuffer::PrintHeader()
{

  unsigned int size = getSize()/sizeof(float);

  cout << _name << "\t " << size << " floats " << endl;

  return size;
}


unsigned int DumpBuffer::Print()
{
  float * ptr = (float*)getData();
  cout << endl;
  unsigned int size =  PrintHeader();
  cout << "-------------------------------------------" << endl; 
  for(int i=0; i<size; i++){
    cout << ptr[i] << endl;
  }

  return size;
}

////////////////////////////////////////////////////////////////////////




DumpFile::~DumpFile()
{

  for ( int i=_buffers.size()-1  ; i >= 0; i--){
    delete _buffers[i];
    _buffers.pop_back();
  }

}

int DumpFile::addBuffer(DumpBuffer * databuffer)
{
  _buffers.push_back(databuffer);

  return _buffers.size()-1;
}

int DumpFile::writeToFile(const char * filename)
{
   ofstream myFile (filename, ios::binary);

  if(!myFile){
    cerr << "file " << filename << " could not be opened!" << endl;
    exit (-1);
  }

  for(int i = 0; i < _buffers.size(); i++){
    _buffers[i]->writeToFile(myFile);;
  }

  myFile.close();

  return _buffers.size(); 
 
}

int DumpFile::readFromFile(const char * filename)
{
  ifstream myFile (filename, ios::binary);

  if(!myFile){
    cerr << "file " << filename << " could not be opened!" << endl;
    exit (-1);
  }
  cout << "reading buffers from " << filename << endl;
  while(myFile.peek() != EOF){
    DumpBuffer* temp = new DumpBuffer(myFile);   
    _buffers.push_back(temp);
  }

  myFile.close();

  return _buffers.size(); 

}

int DumpFile::getNumBuffers()
{
  return _buffers.size();
}


DumpBuffer* DumpFile::getBuffer(int id)
{
  if(id >=0 && id < _buffers.size()) return _buffers[id];
  
  return NULL;
}

DumpBuffer* DumpFile::getBuffer(const char * buffername)
{
  for(int b=0; b<_buffers.size();b++){
    if (0 == strcmp(buffername, _buffers[b]->getName())) return _buffers[b];
  }
  return NULL;
}


//Comparea all the buffers of two DumpFile Objects
void DumpFile::Compare(DumpFile& DF, float threshold, OutputFormat output)
{
  for(int b=0; b<DF._buffers.size();b++){
    Compare(*DF._buffers[b],threshold, output);
  }
}

//compares all the buffers of the DumpFile Object with the passed buffer
void DumpFile::Compare(DumpBuffer& buffer, float threshold, OutputFormat output)
{
  for(int a=0; a<_buffers.size();a++){
    if(_buffers[a]->CompareName(buffer))
      _buffers[a]->Compare(buffer,threshold, output);
  }
}


void DumpFile::printContent()
{
  unsigned int size = 0;
  int cnt = 0;
   for(int a=0; a<_buffers.size();a++){
    if(_buffers[a] != NULL){
      size += _buffers[a]->PrintHeader();
      cnt ++;
    }
  }
  cout <<"----------------------------------------" << endl;
  cout << cnt << " buffers containing: " << size << " floats" << endl << endl;
} 

void DumpFile::printContent(int id)
{
  DumpBuffer * tmp = getBuffer(id);
  if(tmp != NULL) tmp->Print();
}

void DumpFile::printContent(const char * buffername)
{
  DumpBuffer * tmp = getBuffer(buffername);
  if(tmp != NULL) tmp->Print();
} 





