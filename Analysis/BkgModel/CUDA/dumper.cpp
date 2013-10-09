/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "cuda_runtime.h"

#include "dumper.h"


map<DataType, string> DumpBuffer::typeNames = DumpBuffer::creatTypeMap();

DumpBuffer::DumpBuffer()
{
  _buffer =NULL;
  _sizeBytes=0;
  _writeOffset = 0;
  _dtype = determineType();  
  _name[0] = '\0';
  _externData =false;
}


DumpBuffer::DumpBuffer(size_t bytes, const char * name)
{
  _buffer = new char[bytes];
  _sizeBytes=bytes;
  _writeOffset = 0;
 
  _dtype = determineType();  
 
  if(name != NULL) strcpy(_name, name);
  else _name[0] = '\0';

  _externData =false;

}

DumpBuffer::DumpBuffer(ifstream& myFile)
{
  _buffer = NULL;
  _sizeBytes=0;
  _writeOffset = 0;
  _name[0] = '\0';

  _dtype = determineType();  
  
  readFromFile(myFile);

  _externData =false;

}

DumpBuffer::DumpBuffer( DumpBuffer& other, WrapType type)
{
    strcpy(_name, other._name);
    _sizeBytes = other._sizeBytes;
    _writeOffset = other._writeOffset;
    _dtype = other._dtype;
    switch(type){
      case LINK: // link to extern data
        _buffer = other._buffer;
        _externData = true;
        break;
      case LOCAL: // take over extern data and 
        _buffer = other._buffer;
        if(other._externData == false){
          _externData = false;
          other.externData(true); 
        }else{
          _externData = true;
        }
        break;
      case COPY:
      default:
        _buffer = new char[_sizeBytes];
        copy(other._buffer, other._buffer + _sizeBytes, _buffer);
        _externData = false;
    }
}

DumpBuffer & DumpBuffer::operator= (const DumpBuffer & other)
{

  if(this != &other)
  {
    strcpy(_name, other._name);
    _sizeBytes = other._sizeBytes;
    _writeOffset =   other._writeOffset;
    _dtype = other._dtype;
    if(_buffer != NULL && _externData == false) delete [] _buffer;
    _buffer = new char[_sizeBytes];
    copy(other._buffer, other._buffer + _sizeBytes, _buffer);
  }
  return *this; 
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
  if (_buffer != NULL && _externData == false) delete [] _buffer;
  _buffer = NULL;
  _externData = false;
}

void DumpBuffer::externData(bool b)
{
  _externData=b;
}

DataType DumpBuffer::determineType()
{
  return T_FLOAT;
}

size_t DumpBuffer::addData(void *data, size_t bytes)
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
/*
size_t DumpBuffer::addCudaData(void *devData, size_t bytes)
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
*/
bool DumpBuffer::CompareData(DumpBuffer& buffer, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{
  return CompareData((void*)buffer.getData(), threshold, output, length ,stride ,start);
}


bool DumpBuffer::CompareData(void * data, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{
  size_t size = getSize(); 
  char * ptrA = (char*)getData();
  char * ptrB = (char*)data;
  size_t n = 0;
  double stddev = 0;
  size_t cnt = 0;   
  size_t i = start;
  size_t l = 0;
  size_t block = 0;
 
  if(length > stride) size = ( length+start <= size)?(length+start):(size);
  
  while(i < size){


    char diff = abs(ptrA[i] - ptrB[i]);
    double error = abs((double)diff/((ptrA[i] != 0)?(ptrA[i]):(1)));
    if(error>threshold || error != error ){//&& diff > 0.01 ){ 
     if(output > MIN) cout << i <<" " <<block << " " << l <<" " <<  ptrA[i] << " " << ptrB[i] << " " <<  (100.0*error) << "%" << " F" << endl;  
     n++;
     if(error == error) stddev += diff*diff; 
    }else{
     if(output==ALL) cout << i << " " <<block << " " << l <<" " <<  ptrA[i] << " " << ptrB[i] << " " <<  (100.0*error) << "%" << endl;  
     stddev += diff*diff; 
    }

    cnt++; 
    l++;

    if(length == l){
      l = 0;
      block ++;
      if(stride > length){
        i += stride - length; 
//        cout << endl;   
      }
      if(stride == length){
//        cout << endl;
      } 
    }

    i++;
  
  }
  
  stddev =   sqrt(stddev/cnt); 
  
   cout << "compare " << getName() <<" as " << getTypeString() <<  " relative error threshold of " << threshold  << " failed for "<< n << " out of " << cnt << " ( " << (100.0 *n)/((double)(cnt))  << "% ), standard deviation: "<< stddev << endl;
  
  if(n==0){ 
    cout << _name << " correctness check PASSED" << endl;  
    return true; 
  }

  return false;
}

bool DumpBuffer::Compare(DumpBuffer& buffer, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{


  if(!CompareName(buffer)) return false;
  if(!CompareSize(buffer)){ 
    cout << "Buffer " << getName() << " differ in size: " << getSize() << " / " << buffer.getSize() << " Bytes" <<  endl;
    return false;
  }
 
  
  return CompareData((void*)buffer.getData(), threshold, output, length ,stride ,start);
  
}
/*
bool DumpBuffer::CompareCuda(float * devData, float threshold, OutputFormat output)
{
  
  DumpBuffer temp( getSize(), getName());
  temp.addCudaData(devData,getSize());
  return Compare(temp, threshold, output);
} 
*/
void DumpBuffer::writeToFile(ofstream& myFile)
{
  size_t prefix = 0;
  size_t version = DUMP_VERSION;
  // version 2 additional header
  myFile.write((const char*)&prefix , sizeof(size_t));
  myFile.write((const char*)&version , sizeof(size_t));
  myFile.write((const char*)&_dtype, sizeof(DataType));
  // original data in version 1
  myFile.write((const char*)&_writeOffset , sizeof(size_t));
  myFile.write((const char*)_name , sizeof(char)*DUMPNAMELEN);
  myFile.write((const char*)_buffer , _writeOffset);

}
 
void DumpBuffer::readFromFile(ifstream& myFile)
{

 
  cleanup();
  
  size_t version = 1;

  myFile.read((char*)&_sizeBytes,sizeof(size_t));

  if(_sizeBytes == 0) // version 1 no 0 prefix so prefix is _sizeBytes
    myFile.read((char*)&version,sizeof(size_t));

  if(version > 1){
    myFile.read((char*)&_dtype, sizeof(DataType));
    myFile.read((char*)&_sizeBytes,sizeof(size_t));
  }  

  myFile.read((char*)_name,sizeof(char)*DUMPNAMELEN);

  //create and read actual buffer
  _buffer = new char[_sizeBytes];
  myFile.read((char*)_buffer, _sizeBytes);

}


void * DumpBuffer::getData()
{
  return (void *)_buffer; 
}


size_t DumpBuffer::getSize()
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


DataType DumpBuffer::getTypeInfo()
{
  return _dtype;
}

string DumpBuffer::getTypeString()
{
  return typeNames[_dtype]; 
}


size_t DumpBuffer::PrintHeader()
{

  size_t size = getSize();

  cout << _name << "\t " << size <<  " Bytes of type " << getTypeString() << " (Buffer not tempalted)"<< endl;

  return size;
}

size_t DumpBuffer::Print()
{
  cout << endl;
  size_t size =  PrintHeader();
  cout << "Buffer not templated to allow formatted output" << endl;
  return size;
}


size_t DumpBuffer::Print(size_t length , size_t stride ,size_t start, size_t max)
{
  size_t size = getSize(); 
  (void) length;
  (void) stride;
  (void) start;
  (void) max;
  cout << endl;
  PrintHeader();
  cout << "Buffer not templated to allow formatted output" << endl;
  return size; 
}

void DumpBuffer::moveToOtherAndClear(DumpBuffer & other)
{
  
    strcpy(other._name, _name);
    other._sizeBytes = _sizeBytes;
    other._writeOffset = _writeOffset;
    other._dtype = _dtype;
    other._buffer = _buffer;

    _name[0] = '\0';
    _writeOffset = 0;
    _sizeBytes = 0;
    _buffer = NULL;

}


//////////////////////////////////////////////////////////////////////////////////////
// Tempalted wrapper class
//

template<typename T>
DumpWrapper<T>::DumpWrapper(size_t sizeBytes, const char * name ): DumpBuffer(sizeBytes,name) 
{
  _dtype = determineType();  
}

template<typename T>
DumpWrapper<T>::DumpWrapper(ifstream& myFile): DumpBuffer(myFile)
{
  _dtype = determineType();  

}

template<typename T>
DumpWrapper<T>::DumpWrapper(DumpBuffer& other, WrapType type ): DumpBuffer(other, type)
{
  _dtype = determineType();   
}

/*
template<typename T>
DumpWrapper<T>::DumpWrapper(DumpBuffer** other)
{

  (*other)->moveToOtherAndClear(*this);

  _dtype = determineType();
    
    *other = this;
  
}
*/
/*
template <typename T>
DumpWrapper& DumpWrapper<T>::operator=(const DumpBuffer & other)
{


  return *this; 
}

template <typename T>
DumpWrapper& DumpWrapper<T>::operator=(const DumpBuffer & other)
{

  if(this != &other)
  {
    strcpy(_name, other._name);
    _sizeBytes = other._sizeBytes;
    _writeOffset =   other._writeOffset;
    _dtype = determineType();
    if(_buffer != NULL) delete [] _buffer;
    _buffer = new char[_sizeBytes];
    copy(other._buffer, other._buffer + _sizeBytes, _buffer);

  }
  return *this; 
}
*/

template<typename T>
DataType DumpWrapper<T>::determineType()
{

  if(typeid(T) == typeid(char)) return T_CHAR;
  if(typeid(T) == typeid(short)) return T_SHORT;
  if(typeid(T) == typeid(unsigned short)) return T_USHORT;
  if(typeid(T) == typeid(int)) return T_INT;
  if(typeid(T) == typeid(unsigned int)) return T_UINT;
  if(typeid(T) == typeid(float)) return T_FLOAT;
  if(typeid(T) == typeid(double)) return T_DOUBLE;

  return T_UNDEFINED;
}


template<typename T>
size_t DumpWrapper<T>::addData(T *data, size_t bytes)
{
 return DumpBuffer::addData((void*)data, bytes);
}


template<typename T>
bool DumpWrapper<T>::CompareData(DumpBuffer& buffer, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{
  return CompareData((void*)buffer.getData(), threshold, output, length ,stride ,start);
}

template<typename T>
bool DumpWrapper<T>::CompareData(void * data, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{
  size_t size = getSize()/sizeof(T); 
  T * ptrA = (T*)getData();
  T * ptrB = (T*)data;
  size_t n = 0;
  double stddev = 0;
  size_t cnt = 0;   
  size_t i = start;
  size_t l = 0;
  size_t block = 0;

  if(length > stride) size = ( length+start <= size)?(length+start):(size);
  
  while(i < size){


    T diff = abs(ptrA[i] - ptrB[i]);
    double error = abs((double)diff/((ptrA[i] != 0)?(ptrA[i]):(1)));
    if(error>threshold || error != error ){//&& diff > 0.01 ){ 
     if(output > MIN) cout << i <<" " <<block << " " << l <<" " <<  ptrA[i] << " " << ptrB[i] << " " <<  (100.0*error) << "%" << " F" << endl;  
     n++;
     if(error == error) stddev += diff*diff; 
    }else{
     if(output==ALL) cout << i << " " <<block << " " << l <<" " <<  ptrA[i] << " " << ptrB[i] << " " <<  (100.0*error) << "%" << endl;  
     stddev += diff*diff; 
    }

    cnt++; 
    l++;

    if(length == l){
      l = 0;
      block ++;
      if(stride > length){
        i += stride - length; 
//        cout << endl;   
      }
      if(stride == length){
//        cout << endl;
      } 
    }

    i++;
  
  }
  
  stddev =   sqrt(stddev/cnt); 
  
   cout << "compare " << getName()<< " as " << getTypeString() << " relative error threshold of " << threshold  << " failed for "<< n << " out of " << cnt << " ( " << (100.0 *n)/((double)(cnt))  << "% ), standard deviation: "<< stddev << endl;
  
  if(n==0){ 
    cout << _name << " correctness check PASSED" << endl;  
    return true; 
  }

  return false;
}

template<typename T>
bool DumpWrapper<T>::Compare(DumpBuffer& buffer, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{


  if(!CompareName(buffer)) return false;
  if(!CompareSize(buffer)){ 
    cout << "Buffer " << getName() << " differ in number of elements: " << getSize()/sizeof(T) << " / " << buffer.getSize()/sizeof(T) <<   endl;
    return false;
  }
 
  
  return CompareData((void*)buffer.getData(), threshold, output, length ,stride ,start);
  
}



/*
bool DumpWrapper<T>::CompareCuda(float * devData, float * hostData, size_t size, float threshold, OutputFormat output)
{
  DumpBuffer tmpCuda(size,"CudaBuffer");
  tmpCuda.addCudaData(devData,size);
  return tmpCuda.CompareData(hostData,threshold, output);
}

*/


template<typename T>
size_t DumpWrapper<T>::PrintHeader()
{

  size_t size = getSize()/sizeof(T);

  cout << _name << "\t " << size <<  " elements of type " << getTypeString() << endl;

  return size;
}

template <typename T>
size_t DumpWrapper<T>::Print()
{
  T * ptr = (T*)getData();
  cout << endl;
  size_t size =  PrintHeader();
  cout << "-------------------------------------------" << endl; 
  for(size_t i=0; i<size; i++){
    cout << ptr[i] << endl;
  }

  return size;
}


template<typename T>
size_t DumpWrapper<T>::Print(size_t length , size_t stride ,size_t start, size_t max)
{
  size_t size = getSize()/sizeof(T); 
  T * ptrA = (T*)getData();
  size_t cnt = 0;   
  size_t i = start;
  size_t l = 0;
  size_t block = 0;

  PrintHeader();
  cout << "-------------------------------------------" << endl; 

  if (max == 0) max = size;
 
  if(length > stride) size = ( length+start <= size)?(length+start):(size);
  cout << "absolut_element stride_block offset_in_block value" <<endl; 
  while(i < size && i < max){
    cout << i << " " <<block << " " << l <<" " <<  ptrA[i] << endl;  

    cnt++; 
    l++;
   
    if(length == l){
      l = 0;
      block ++;
      if(stride > length){
        i += stride - length; 
//        cout << endl;   
      }
      if(stride == length){
//        cout << endl;
      } 
    }
    i++;
  }
  return size;
}




////////////////////////////////////////////////////////////////////////




DumpFile::~DumpFile()
{
//  vector<DumpBuffer*>::iterator it;
//  for ( it = _buffers.begin(); it != _buffers.end(); ){
//    delete * it;  
//    it = Entities.erase(it);
//  }

  for ( size_t i=_buffers.size(); i > 0; i--){
    if(_buffers[i-1] != NULL) delete _buffers[i-1];
    _buffers.pop_back();
  }

}

size_t DumpFile::addBuffer(DumpBuffer * databuffer)
{
  _buffers.push_back(databuffer);
  return _buffers.size()-1;
}

size_t DumpFile::writeToFile(const char * filename)
{
   ofstream myFile (filename, ios::binary);

  if(!myFile){
    cerr << "file " << filename << " could not be opened!" << endl;
    exit (-1);
  }

  for(size_t i = 0; i < _buffers.size(); i++){
    _buffers[i]->writeToFile(myFile);;
  }

  myFile.close();

  return _buffers.size(); 
 
}

size_t DumpFile::readFromFile(const char * filename)
{
  ifstream myFile (filename, ios::binary);

  cout << endl << "Reading buffers from file: " << filename << endl;
  cout <<"----------------------------------------" << endl;

  if(!myFile){
    cerr << "file " << filename << " could not be opened!" << endl;
    exit (-1);
  }
  while(myFile.peek() != EOF){
    DumpBuffer* baseptr = new DumpBuffer(myFile);   
    DataType ti = baseptr->getTypeInfo();
    DumpBuffer* temp;

    switch(ti){
      case T_INT:
        temp = new DumpWrapper<int>( *baseptr, LOCAL );
        break;
      case T_UINT:
        temp = new DumpWrapper<unsigned int>( *baseptr, LOCAL );
        break;
      case T_SHORT:
        temp = new DumpWrapper<short>( *baseptr, LOCAL );
        break;
      case T_USHORT:
        temp = new DumpWrapper<unsigned short>( *baseptr, LOCAL );
        break;
          case T_DOUBLE:
        temp = new DumpWrapper<double>( *baseptr, LOCAL );
        break;
      case T_CHAR:
        temp = new DumpWrapper<char>( *baseptr, LOCAL );
        break;
      case T_FLOAT:
      default:
        temp = new DumpWrapper<float>( *baseptr, LOCAL );
        break;
    }
    addBuffer(temp);
    delete baseptr; 
  }


  myFile.close();

  return _buffers.size(); 

}

size_t DumpFile::getNumBuffers()
{
  return _buffers.size();
}


DumpBuffer* DumpFile::getBuffer(unsigned int id)
{
  if( id < _buffers.size()) return _buffers[id];
  
  return NULL;
}

DumpBuffer* DumpFile::getBuffer(const char * buffername)
{
  for(size_t b=0; b<_buffers.size();b++){
    if (0 == strcmp(buffername, _buffers[b]->getName())) return _buffers[b];
  }
  return NULL;
}


//Comparea all the buffers of two DumpFile Objects
void DumpFile::Compare(DumpFile& DF, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{
  for(size_t b=0; b<DF._buffers.size();b++){
    Compare(*DF._buffers[b],threshold, output, length ,stride ,start);
  }
}

//compares all the buffers of the DumpFile Object with the passed buffer
void DumpFile::Compare(DumpBuffer& buffer, float threshold, OutputFormat output, size_t length , size_t stride ,size_t start)
{
  for(size_t a=0; a<_buffers.size();a++){
    if(_buffers[a]->CompareName(buffer))
      _buffers[a]->Compare(buffer,threshold, output, length ,stride ,start);
  }
}

void DumpFile::printContent()
{
  size_t size = 0;
  size_t cnt = 0;
   for(size_t a=0; a<_buffers.size();a++){
    if(_buffers[a] != NULL){
      size += _buffers[a]->PrintHeader();
      cnt ++;
    }
  }
  cout <<"----------------------------------------" << endl;
  cout << cnt << " buffers containing: " << size << " data points" << endl;
  cout <<"----------------------------------------" << endl;

} 

void DumpFile::printContent(int id, size_t length , size_t stride ,size_t start)
{
  DumpBuffer * tmp = getBuffer(id);
  if(tmp != NULL) tmp->Print(length,stride,start);
}

void DumpFile::printContent(const char * buffername, size_t length , size_t stride ,size_t start)
{
  DumpBuffer * tmp = getBuffer(buffername);
  if(tmp != NULL) tmp->Print(length,stride,start);
} 

// explicit compile of templates

template class DumpWrapper<char>;
template class DumpWrapper<short>;
template class DumpWrapper<unsigned short>;
template class DumpWrapper<int>;
template class DumpWrapper<unsigned int>;
template class DumpWrapper<float>;
template class DumpWrapper<double>;




