/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * LayoutCubeTemplate.h
 *
 *  Created on: Oct 30, 2014
 *      Author: Jakob Siegel
 */

#ifndef LAYOUTCUBETEMPLATE_H_
#define LAYOUTCUBETEMPLATE_H_

#include "MemoryManager.h"
#include "ImgRegParams.h"



#define DEFAULT_MEMORY_TYPE HostMem

////////////////////////////////////////////////////
//class LayoutCube
//creates a data cube and allocates it in page-locked host memory
//offers access to all elements through put and get methods
//using x,y,z coordinates
//also offers consecutive access through a readWritePointer
//and a stride. with every read or write the RWptr gets moved by
//the stride
template<typename T>
class LayoutCube : public TMemSegAlloc<T>
{

protected:

  dim3 _dim;
  size_t _accessAt;
  size_t _stride;

  T* getRWPtr(){ return (this->getPtr()) + _accessAt; }

public:
  //no allocation just wrapp around already allocated buffer, size of buffer has to be >= x*y*z*sizeof(T)
  LayoutCube(T *ptr, size_t x, size_t y=1, size_t z=1, MemoryType type = DEFAULT_MEMORY_TYPE):
    TMemSegAlloc<T>(ptr, x*y*z*sizeof(T), type), _dim(x,y,z), _accessAt(0), _stride(x*y) {};

  //allocation
  LayoutCube(size_t x, size_t y=1, size_t z=1, MemoryType mtype = DEFAULT_MEMORY_TYPE):
    TMemSegAlloc<T>(x*y*z*sizeof(T), mtype),  _dim(x,y,z), _accessAt(0), _stride(x*y) {};

  //physically copy Buffer MemSegment to a buffer of mtype and wrap it into a LayoutCube
  LayoutCube(LayoutCube<T> & that, MemoryType mtype): TMemSegAlloc<T>(that, mtype), _dim(that._dim), _accessAt(0), _stride(that.getRWStrideZ()) {};

  virtual ~LayoutCube(){};

  void wrappPtr(T * ptr, size_t numElements = 0, MemoryType mtype = UnDefined);


  bool checkCoords (size_t x, size_t y, size_t z) const;
  bool checkIdx (size_t idx) const;
  bool checkAccess(size_t idx) const;
  //checks current RW pointer accessibility
  bool checkAccess();

  inline void moveByStride() { _accessAt += _stride; }
  //stream like, read write pointer(RWPtr) access, moves by stride after read or write
  //sets RWPtr to beginning of buffer
  void resetRWPtr(){ _accessAt = 0; }
  //sets RWPtr to provided location within data cube
  void setRWPtr(size_t x,size_t y=0,size_t z=0) {  checkCoords(x,y,z);  _accessAt = z*_dim.x*_dim.y + y*_dim.x + x;  }
  void setRWPtrIdx(size_t idx, size_t plane = 0) {  size_t readIdx = idx + plane * getRWStrideZ(); checkIdx(readIdx); _accessAt = readIdx;}
  size_t getRWlocXYIdx(){return _accessAt%(_dim.x*_dim.y);} // Id in XY plane
  size_t getRWlocX(){return _accessAt%_dim.x;} // X offset
  size_t getRWlocY(){return  getRWlocXYIdx()/_dim.x;} // Y offset
  size_t getRWlocZ(){return _accessAt/(_dim.x*_dim.y);} //Z offset
  //allows to set a custom stride to applied to RWPtr after read or write
  void setRWStride(size_t stride){_stride = stride;}
  size_t getRWStride(){return _stride;}
  //sets predefined stride to advance through buffer cube in X,Y or Z direction
  void setRWStrideX(){_stride = 1;}
  size_t getRWStrideX(){return 1;}
  //sets predefined stride to advance through buffer cube in X,Y or Z direction
  void setRWStrideY(){_stride = _dim.x;}
  size_t getRWStrideY(){return _dim.x;}
  //sets predefined stride to advance through buffer cube in X,Y or Z direction
  void setRWStrideZ(){_stride = _dim.x*_dim.y;}
  size_t getRWStrideZ(){return _dim.x*_dim.y;}
  size_t getStrideLimit(){
    if(_stride == getRWStrideX() ) return _dim.x;
    if(_stride == getRWStrideY() ) return _dim.y;
    if(_stride == getRWStrideZ() ) return _dim.z;
    return this->getNumElements()/_stride;
  }

  //faster, stream like access, through RWPtr
  //reads or writes like the buffer is a stream and advances the RWPtr by stride
  void write(T value){ checkAccess();  (*this)[_accessAt] = value;  moveByStride(); }
  T read(){ checkAccess(); T val = (*this)[_accessAt]; moveByStride(); return val; }
  //writes num elements along the set stride starting at RWPtr and advances the pointer bei num * stride
  void writeByStride( T* buffer, size_t num){ for(size_t i=0; i < num; i++) write(buffer[i]); }

  //access via absolute offset, sets and advances RW pointer by stride
  void putAt(T value,size_t x,size_t y=0,size_t z=0){ setRWPtr(x,y,z); write(value); };

  //const access functions
  T getAt(size_t x,size_t y=0,size_t z=0) const;
  T getAtIdx(size_t idx, size_t plane = 0)const;

  //returns a reference to the element the RWpointer points at and advances RW pointer by stride
  T & ref(){ checkAccess(); size_t at = _accessAt; moveByStride() ; return (*this)[at];}
  //returns a reference to the element at x,y,z, sets and advances RW pointer by stride
  T & refAt(size_t x,size_t y=0,size_t z=0){ checkCoords(x,y,z);  _accessAt = z*_dim.x*_dim.y + y*_dim.x + x; checkAccess(); return ref();}

  //T& operator[](size_t idx){
  //  return TMemSegAlloc<T>::[idx];
  //}


  //only works on host side
  void SetValue(T value, int plane = -1);

  //copies a subset of planes from src to this (dst)
  //if numPlanes 0 is provided all the planes from in the range from srcPlane to the end of the src segment are copied.
  void copyPlanes( LayoutCube<T> & src, size_t srcPlane, size_t dstPlane, size_t numPlanes = 0);
  //copies numPlanes from src to dstPlane
  void copyPlanesIn( size_t dstPlane, T * src, size_t numPlanes = 1);
  //copies numPlanes starting with srcPlane to dts buffer
  void copyPlanesOut( T * dst, size_t srcPlane, size_t numPlanes = 1);

  //works on host and device side
  void memSetPlane( int value, size_t plane, size_t numPlanes = 1);
  //like read or write but pointer is not moved
  void putCurrent(T value){  checkAccess(); (*this)[_accessAt] = value;}
  T getCurrent(){ checkAccess(); T val = (*this)[_accessAt]; return val; }

  T * getPtrToPlane(int plane);

  virtual ImgRegParams getParams() const ;

  size_t getDimX() const {return _dim.x; };
  size_t getDimY() const  {return _dim.y; };
  size_t getDimZ() const  {return _dim.z; };
  dim3 getDims() const  { return _dim; }



  void dumpCubeToFile(ofstream & myFile);
  bool readCubeFromFile(ifstream & myFile);
  void print() const ;

  static size_t calculateSize( size_t x, size_t y, size_t z) { return x*y*z*sizeof(T);}

private:
  LayoutCube & operator=(const LayoutCube & that); //disallow assignment operator

};

///////////////////////////////////////////////////////////////////////////////////////
/////


//Layout Cube:

template<typename T>
void LayoutCube<T>::wrappPtr(T * ptr, size_t numElements, MemoryType mtype)
{
  MemoryType thisType = (mtype != UnDefined)?(mtype):(this->getType());
  if(thisType == UnDefined) thisType = HostMem;
  if(numElements  == 0) numElements = getDimX() * getDimY() * getDimZ();
  size_t sizeBytes = numElements * sizeof(T);
  TMemSegAlloc<T>::wrapp(ptr,sizeBytes,thisType);
}




template<typename T>
bool LayoutCube<T>::checkCoords(size_t x, size_t y, size_t z) const {
  if( x < _dim.x && y < _dim.y && z < _dim.z) return true;
  cerr << "provided coordinates (" << x << "," << y << "," << z << ") are out of bounds and will cause segfault if accessed" << endl;
  assert(x < _dim.x && y < _dim.y && z < _dim.z);
  return false;
}

template<typename T>
bool LayoutCube<T>::checkIdx(size_t idx) const{
  size_t sizeT = this->getNumElements();
  if(idx >= sizeT){
    cerr << "access out of bounds trying to access element ["<< _accessAt << "] in buffer of size " << this->getNumElements() << endl;
    return false;
  }
  return true;
}

template<typename T>
bool LayoutCube<T>::checkAccess(){
  if(this->getType() == DeviceGlobal){
    cerr << "access of device buffer on host side not possible" << endl;
    exit(-1);
    //return false;
  }
  return (checkIdx(_accessAt));
}
template<typename T>
bool LayoutCube<T>::checkAccess(size_t idx) const {
  if(this->getType() == DeviceGlobal){
    cerr << "access of device buffer on host side not possible" << endl;
    exit(-1);
    //return false;
  }
  return (checkIdx(_accessAt));
}

template<typename T>
T LayoutCube<T>::getAt(size_t x,size_t y,size_t z) const {
  checkCoords(x,y,z);
  size_t offset = z*_dim.x*_dim.y + y*_dim.x + x;
  return this->getElement(offset);
}
template<typename T>
T LayoutCube<T>::getAtIdx(size_t idx, size_t plane)const
{
  size_t offset = plane *_dim.x*_dim.y + idx;
  checkIdx(offset);
  return this->getElement(offset);
}



//only works on host side
template<typename T>
void LayoutCube<T>::SetValue(T value, int plane){
  size_t start = 0;
  size_t end = this->getNumElements();
  if(plane >= 0 ){
    start = getRWStrideZ()*plane;
    end = (start + getRWStrideZ() < end)?(start + getRWStrideZ()):(end);
  }
  size_t oldStride = getRWStride();
  setRWStrideX();
  setRWPtrIdx(start);
  for(size_t i = start; i < end; i++)
    write(value);
  setRWStride(oldStride);
}

template<typename T>
void LayoutCube<T>::copyPlanes( LayoutCube<T> & src, size_t srcPlane, size_t dstPlane, size_t numPlanes)
{
  //get byte offset
  srcPlane *= src.getRWStrideZ() * sizeof(T);
  dstPlane *= getRWStrideZ() * sizeof(T);
  //get size in bytes
  numPlanes *= getRWStrideZ() * sizeof(T);
  this->copySubSet(src, srcPlane, dstPlane, numPlanes);
}

template<typename T>
void LayoutCube<T>:: copyPlanesIn( size_t dstPlane, T * src, size_t numPlanes)
{
  size_t sizePlaneBytes = getRWStrideZ() * sizeof(T);
  size_t dstPlaneOffset = dstPlane* sizePlaneBytes;
  size_t sizeBytes = numPlanes * sizePlaneBytes;
  this->copyInSubSet(dstPlaneOffset,src,sizeBytes);
}

//copies numPlanes (default 1) starting with srcPlane to dts buffer
template<typename T>
void LayoutCube<T>:: copyPlanesOut( T * dst, size_t srcPlane, size_t numPlanes )
{
  size_t sizePlaneBytes = getRWStrideZ() * sizeof(T);
  size_t dstPlaneOffset = srcPlane* sizePlaneBytes;
  size_t sizeBytes = numPlanes * sizePlaneBytes;
  this->copyOutSubSet(dst,dstPlaneOffset,sizeBytes);
}



template<typename T>
void LayoutCube<T>::memSetPlane( int value, size_t plane, size_t numPlanes)
{
  size_t planeSizeBytes = getRWStrideZ() * sizeof(T);
  size_t offsetBytes = planeSizeBytes * plane;
  this->memSet(value,offsetBytes,planeSizeBytes * numPlanes);
}


template<typename T>
T * LayoutCube<T>::getPtrToPlane(int plane){
  T * ptr = this->getPtr();
  ptr += getRWStrideZ()*plane;
  return ptr;
}

template<typename T>
ImgRegParams LayoutCube<T>::getParams() const
{
  ImgRegParams tmp;
  tmp.init(this->getDimX(), this->getDimY(), this->getDimX(), this->getDimY());
  return tmp;
}

template<typename T>
void LayoutCube<T>::dumpCubeToFile(ofstream & myFile)
{
  ImgRegParams irp = getParams();
  myFile.write((const char*)&irp,sizeof(ImgRegParams));
  size_t dimz = getDimZ();
  myFile.write((const char*)&dimz,sizeof(size_t));
  this->dumptoFile(myFile);
}

template<typename T>
bool LayoutCube<T>::readCubeFromFile(ifstream & myFile)
{

  ImgRegParams irtmp;
  myFile.read((char*)&irtmp,sizeof(ImgRegParams));
  size_t dimz;
  myFile.read((char*)&dimz,sizeof(size_t));
  if(irtmp == getParams() && dimz == getDimZ()){
    this->readFromFile(myFile);
  }else{
    cout << "Could not read Buffer from file Image dimensions missmatch!" << endl;
    cout << "buffer dimensions: depth "<< getDimZ() <<" "; getParams().print();
    cout << "file dimensions: depth "<< dimz <<" ";  irtmp.print();
    return false;
  }
  return true;
}

template<typename T>
void LayoutCube<T>::print() const {
  cout << "Cube of type " << this->getType() <<  ": " << this->getDimX() << ":" << getDimY() << ":" << getDimZ() << " size: " << this->getNumElements() << "(" << this->getSize() << "bytes)"  << endl;
}



#endif /* LAYOUTCUBETEMPLATE_H_ */
