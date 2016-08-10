/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * LayoutCubeRegionsTemplate.h
 *
 *  Created on: Oct 30, 2014
 *      Author: Jakob Siegel
 */

#ifndef LAYOUTCUBEREGIONSTEMPLATE_H_
#define LAYOUTCUBEREGIONSTEMPLATE_H_

#include "LayoutCubeTemplate.h"
#include "NonBasicCompare.h"



////////////////////////////////////
//class LayoutCubeWithRegions
//projects a grid of regions on top the x/y plane of the original cube
//adds functionality to access data inside a region by region id and the local x and y
template<typename T>
class LayoutCubeWithRegions : public LayoutCube<T>
{

protected:

  size_t _regWidth;
  size_t _regHeight;

public:

  //Wrapper Constructor
  LayoutCubeWithRegions(T *ptr, size_t imgWidth, size_t imgHeight = 1 , size_t maxRegWidth = 1, size_t maxRegHeight = 1, size_t numPlanes = 1, MemoryType type = DEFAULT_MEMORY_TYPE ):
    LayoutCube<T>(ptr,imgWidth,imgHeight,numPlanes,type),_regWidth(maxRegWidth),_regHeight(maxRegHeight)
    {};

  LayoutCubeWithRegions(T *ptr, ImgRegParams iP, size_t numPlanes = 1, MemoryType type = DEFAULT_MEMORY_TYPE ):
      LayoutCube<T>(ptr,iP.getImgW(),iP.getImgH(),numPlanes,type),_regWidth(iP.getRegW()),_regHeight(iP.getRegH())
      {};

  //Allocator constructors
  LayoutCubeWithRegions(size_t imgWidth, size_t imgHeight = 1 , size_t maxRegWidth = 1, size_t maxRegHeight = 1, size_t numPlanes = 1, MemoryType mtype = DEFAULT_MEMORY_TYPE ):
    LayoutCube<T>(imgWidth,imgHeight,numPlanes,mtype),_regWidth(maxRegWidth),_regHeight(maxRegHeight)
    {};

  LayoutCubeWithRegions(ImgRegParams iP, size_t numPlanes = 1, MemoryType mtype = DEFAULT_MEMORY_TYPE ):
    LayoutCube<T>(iP.getImgW(),iP.getImgH(),numPlanes,mtype),_regWidth(iP.getRegW()),_regHeight(iP.getRegH())
    {};

  LayoutCubeWithRegions(const LayoutCubeWithRegions<T> & that, MemoryType mtype):
    LayoutCube<T>(that, mtype),_regWidth(that.getRegW(0)),_regHeight(that.getRegH(0))
    {};


  //DEBUG CONSTRUCTOR adds actual byte size that was allocated to the passed vector
  LayoutCubeWithRegions(ImgRegParams iP, size_t numPlanes, MemoryType mtype, vector<size_t>& sizeBytes  ):
        LayoutCube<T>(iP.getImgW(),iP.getImgH(),numPlanes,mtype),_regWidth(iP.getRegW()),_regHeight(iP.getRegH())
        {
          sizeBytes.push_back(this->getSize());
        };

  virtual ~LayoutCubeWithRegions(){}


  void setRWPtrRegion(size_t regId, size_t rx=0, size_t ry=0, size_t z =0);
  //returns region id of region where the RW pointer currently is residing
  size_t getRWlocRegId();
  //Returns x offset of write pointer within region
  size_t getRWlocRegX();
  //Returns y offset of write pointer within region
  size_t getRWlocRegY();
  // x and y inside region, sets and advances RW pointer by stride
  T getAtReg(size_t regId, size_t rx=0, size_t ry=0, size_t z=0);
  // x and y inside region, sets and advances RW pointer by stride
  void putAtReg(T value, size_t regId, size_t rx=0, size_t ry=0, size_t z=0);
  //returns a reference to the element at regId,x,y,z, sets and advances RW pointer by stride
  T & refAtReg(size_t regId, size_t rx=0, size_t ry=0, size_t z=0);
  //returns actual region width and height
  size_t getRegW (size_t regId) const;
  size_t getRegH (size_t regId) const;

  virtual ImgRegParams getParams() const;

  void SetValueRegion(T value, size_t regId, int iplane = -1);

  void print() const;

  static size_t calculateSize(ImgRegParams iP, size_t numPlanes) { return iP.getImgW()*iP.getImgH()*numPlanes*sizeof(T);}

  //will write regW*regH elements from data into the cube. write happens row major in designated plane
  //data in source buffer will write n elements (<= numElements in Region) with a stride of 'stride'
  //for any stride >1 you can write nPerStride elements to the region with nPerStride <= stride;
  void putRegData(size_t regId, T * data, size_t n = 0, size_t plane = 0, size_t stride = 1, size_t nPerStride = 1);
  void putRegData(size_t rC, size_t rR, T * data, size_t plane, size_t n, size_t stride = 1, size_t nPerStride = 1);

  //copies region iRegId from input to regId of this cube. will not copy if dimension check fails
  //if CropOrFill is set to true it will ignore dimension check and will copy whatever fits into the destination region or start
  //replicating when destination is larger than source.
  void copyReg(size_t regId, LayoutCubeWithRegions<T> & input, size_t iRegId, size_t numPlanes = 0, size_t startPlane = 0, bool CropOrFill = false);

  //returns CSV string of designated values
  //typename P allows to cast the output to specific type that might be different from T
  //P must provide a << operator
  //regId,x,y,z determine first element of the string
  //limit defines number of elements added to the string
  //if no limit is provided the maximum number of elements in stride direction are added to the string
  //if limit > than stride limit the program will fail with an out of bounds error
  //the series of elements are defined by the set read/Write stride (setRWStride<XYZ>())
  //offsetBytes allows to output a value of type P at a offset of bytes from the base of a value T
  //e.g. to output a specific element from a type T that is a non basic type/structure.
  //struct T { double a, int b, float c), a P of <float> and a byteOffset of 12 will output a CSV list
  //of all float c members of the structures T read from the Cube
  //delimiter by default is ',' but can be set to any char provided

  template <typename P>
  string getCSVatReg(size_t regId, size_t rx=0, size_t ry=0, size_t z=0, size_t limit = 0, size_t offsetBytes = 0, char delimiter = ',');
  //returns string of all <P> values in plane for one region. string has lyout of region and terminates each reagion row with endl
  template<typename P>
  string getCSVRegionPlane(size_t regId, size_t xlimit=0, size_t ylimit=0,  size_t plane = 0, size_t offsetBytes = 0, char delimiter = ',');

  template<typename P>
  string getCSVRegionCube(size_t regId, size_t xlimit=0, size_t ylimit=0, size_t zlimit = 0, size_t offsetBytes = 0, char delimiter = ',');

  //prints a table of the elements at rx:ry:rz of every region casted to type of P which can be different from type T
  //if elements are objects which do not offer a "<<" operator for printing a byte offset within the object to can be provided
  //can be provided.
  template <typename P>
  void printRegionTable( size_t rx = 0, size_t ry = 0, size_t rz = 0, size_t offsetBytes = 0);

  template <typename P>
  float getAvgReg(  size_t regId,
      size_t plane = 0,
      size_t offsetBytes = 0,
      LayoutCubeWithRegions<unsigned short> * mask = NULL,
      unsigned short maskValue = 0
  );



  template <typename P>
  float getStdDevReg( size_t regId,
      size_t plane = 0,
      float avg = 0.0f,
      size_t offsetBytes = 0,
      LayoutCubeWithRegions<unsigned short> * mask = NULL,
      unsigned short maskValue = 0
  );

  //most parameters as above for CSV writer
  // P has to provide << and == or a compare(P,P) function pointer
  //limit will limit the comparison to #limit base elements and
  //num  > 1 will compare num elements of type P starting from offsetBytes from a T element at regId,x,y,z
  //if a mask is provided it will only be appalled to the base coordinates designated by the stride of an entry.
  //the mask will not be evaluated for elements designated by offset or num entries from the base coordinate
  /*template <typename P>
  bool compare( LayoutCubeWithRegions<T> & that,
                size_t regId, size_t x=0, size_t y=0, size_t z=0,
                size_t limit = 0, size_t offsetBytes = 0, size_t num = 1,
                CCompare<P> *comparePtr = NULL,
                LayoutCubeWithRegions<unsigned short> * mask = NULL,
                unsigned short maskValueToMatch = 0,
                CompareOutput print= CompOutAll)*/
  template <typename P>
  bool compare(   LayoutCubeWithRegions<T> & that,
      size_t regId, size_t rx=0, size_t ry=0, size_t z=0,
      size_t limit = 0, size_t offsetBytes = 0, size_t num = 1,
      CCompare<P> *comparePtr = NULL,
      LayoutCubeWithRegions<unsigned short> * mask = NULL,
      unsigned short maskValueToMatch = 0,
      CompareOutput  verbose = CompOutAll
  );


protected:

  bool checkCoords(size_t regIdx, size_t rx, size_t ry, size_t z) const;

  //HELPER:
  //returns number of Row and Columns in the region grid
  size_t gridDimX() const { return ((this->getDimX() +_regWidth -1)/_regWidth);}
  size_t gridDimY() const { return ((this->getDimY() +_regHeight -1)/_regHeight);}

  //return column and row region redId is in
  size_t regCol(size_t regId) const { return regId%gridDimX();}
  size_t regRow(size_t regId) const { return regId/gridDimX();}

  //return x and y coordinate of first region element
  size_t regBaseX(size_t regId) const {  return regCol(regId)*_regWidth; };
  size_t regBaseY(size_t regId) const {  return regRow(regId)*_regHeight; };



};

///////////////////////////////////////////////////////////////////////////////
//////
///// out of class Implementation

template<typename T>
void LayoutCubeWithRegions<T>::setRWPtrRegion(size_t regId, size_t rx, size_t ry, size_t z)
{
  this->checkCoords(regId,rx,ry,z);
  this->setRWPtr(regBaseX(regId)+rx, regBaseY(regId)+ry, z);
}



//returns region id of region where the RW pointer currently is residing
template<typename T>
size_t LayoutCubeWithRegions<T>::getRWlocRegId(){
  size_t ix = this->getRWlocX();
  size_t iy = this->getRWlocY();
  return this->getParams().getRegId(ix,iy);
  //return ((iy/_regHeight) * ((this->getDimX() + _regWidth -1)/_regWidth) + (ix/_regWidth));
}
//retunrs x offset of write pointer within region
template<typename T>
size_t LayoutCubeWithRegions<T>::getRWlocRegX(){
  size_t ix = this->_accessAt% this->getDimX();
  return ix %_regWidth;
}
//retunrs y offset of write pointer within region
template<typename T>
size_t LayoutCubeWithRegions<T>::getRWlocRegY(){
  size_t iy = this->_accessAt/ this->getDimX();
  return iy%_regHeight ;
}

// x and y inside region, sets and advances RW pointer by stride
template<typename T>
T LayoutCubeWithRegions<T>::getAtReg(size_t regId, size_t rx, size_t ry, size_t z){
  setRWPtrRegion(regId,rx,ry,z);
  return this->read();
}
// x and y inside region, sets and advances RW pointer by stride
template<typename T>
void LayoutCubeWithRegions<T>::putAtReg(T value, size_t regId, size_t rx, size_t ry, size_t z){
  setRWPtrRegion(regId,rx,ry,z);
  this->write(value);
}

//returns a reference to the element at regId,x,y,z, sets and advances RW pointer by stride
template<typename T>
T& LayoutCubeWithRegions<T>::refAtReg(size_t regId, size_t rx, size_t ry, size_t z){
  return this->refAt(regBaseX(regId) + rx,regBaseY(regId) + ry,z);
}

//returns actual region width and height
template<typename T>
size_t LayoutCubeWithRegions<T>::getRegW(size_t regId) const { size_t b = this->getDimX()-regBaseX(regId); return ( b > _regWidth)?(_regWidth):(b); }
template<typename T>
size_t LayoutCubeWithRegions<T>::getRegH(size_t regId) const {size_t b = this->getDimY()-regBaseY(regId); return ( b > _regHeight)?(_regHeight):(b); }

template<typename T>
ImgRegParams LayoutCubeWithRegions<T>::getParams() const {
  ImgRegParams tmp(this->getDimX(), this->getDimY(), _regWidth, _regHeight);
  return tmp;
}

template<typename T>
void LayoutCubeWithRegions<T>::SetValueRegion(T value, size_t regId, int iplane){
  size_t oldStride = this->getRWStride();
  size_t start = 0;
  size_t end = this->getDimZ();
  size_t plane = iplane;
  if(iplane >= 0 )
    start = end = plane;
  this->setRWStrideX();
  for(size_t p = start; p < end; p++ ){
    for(size_t ry = 0; ry < getRegH(regId); ry++){
      this->setRWPtrRegion(regId,0,ry,p);
      for(size_t rx = 0; rx < getRegW(regId); rx++){
        this->write(value);
      }
    }
  }
  this->setRWStride(oldStride);
}


template<typename T>
void LayoutCubeWithRegions<T>::print() const {
  cout << "Cube of type " << this->getType() <<  ": " << this->getDimX() << ":" << this->getDimY() << ":" << this->getDimZ() << " size: " << this->getNumElements() << "(" << this->getSize() << "bytes) with regions: " << _regWidth << ":" << _regHeight << " region grid: " << gridDimX() << ":" << gridDimY() << endl;
}



//will write regW*regH elements from data into the cube. write happens row major in designated plane
//data in source buffer will write n elements (<= numElements in Region) with a stride of 'stride'
//for any stride >1 you can write nPerStride elements to the region with nPerStride <= stride;
template<typename T>
void LayoutCubeWithRegions<T>::putRegData(size_t regId, T * data, size_t n, size_t plane, size_t stride , size_t nPerStride ){

  assert( nPerStride <= stride);
  size_t oldStride = this->getRWStride();
  this->setRWStrideX();
  if(n == 0) n = getParams().getRegSize(regId);
  size_t i = 0;
  size_t s = 0;
  size_t offset = 0;
  for(size_t ry=0; (ry < getParams().getRegH(regId)) && (i<n); ry++){
    this->setRWPtrRegion(regId,0,ry,plane);
    size_t rx = 0;
    while( (rx <getParams().getRegW(regId)) && (i<n) ){
      this->write(data[offset+s]);
      i++;
      rx++;
      s++;
      if(s == nPerStride ){
        offset += stride;
        s = 0;
      }
    }
  }
  this->setRWStride(oldStride);
}

template<typename T>
void LayoutCubeWithRegions<T>::putRegData(size_t rC, size_t rR, T * data, size_t plane, size_t n, size_t stride, size_t nPerStride){
  putRegData(getParams().getRegId(rC,rR),data,plane,n,stride,nPerStride );
}


//copies region iRegId from input to regId of this cube. will not copy if dimension check fails
//if CropOrFill is set to true it will ignore dimension check and will copy whatever fits into the destination region or start
//replicating when destination is larger than source.
template<typename T>
void LayoutCubeWithRegions<T>::copyReg(size_t regId, LayoutCubeWithRegions<T> & input, size_t iRegId, size_t numPlanes, size_t startPlane, bool CropOrFill){

  if(numPlanes == 0) numPlanes = this->getDimZ();

  assert( startPlane+numPlanes <= this->getDimZ()); //check for enough planes in destination

  //if we do not replicate or crop check source dimensions match and source contains enough planes
  if(CropOrFill == false){
    assert(this->getRegW(regId) == input.getRegW(iRegId) && this->getRegH(regId) == input.getRegH(iRegId)); //check for identical region size
    assert( startPlane+numPlanes <= input.getDimZ()); //check for enough planes
  }

  input.setRWStrideX();
  this->setRWStrideX();

  for(size_t p=startPlane; p< numPlanes; p++){
    for(size_t ry=0; ry < this->getRegH(regId); ry++){
      input.setRWPtrRegion(iRegId,0,ry%input.getRegH(iRegId), p%input.getDimZ()); //replicate source id boundaries reached
      this->setRWPtrRegion(regId,0,ry,p);
      for(size_t rx=0; rx < this->getRegW(regId); rx++){
        if(input.getRWlocRegX() >= input.getRegW(iRegId)) input.setRWPtrRegion(iRegId,0,ry,p); //replicate soure if boundaries reached
        this->write(input.read());
      }
    }
  }
}

//returns CSV string of designated values
//typename P allows to cast the output to specific type that might be different from T
//P must provide a << operator
//regId,x,y,z determine first element of the string
//limit defines number of elements added to the string
//if no limit is provided the maximum number of elements in stride direction are added to the string
//if limit > than stride limit the program will fail with an out of bounds error
//the series of elements are defined by the set read/Write stride (setRWStride<XYZ>())
//offsetBytes allows to output a value of type P at a offset of bytes from the base of a value T
//e.g. to output a specific element from a type T that is a non basic type/structure.
//struct T { double a, int b, float c), a P of <float> and a byteOffset of 12 will output a CSV list
//of all float c members of the structures T read from the Cube
//delimiter by default is ',' but can be set to any char provided
template<typename T>
template<typename P>
string LayoutCubeWithRegions<T>::getCSVatReg(size_t regId, size_t rx, size_t ry, size_t z, size_t limit, size_t offsetBytes, char delimiter)
{
  ostringstream csvstring;
  P value;
  LayoutCubeWithRegions<T> * me = this;
  if(this->getType() == DeviceGlobal)
    me = new LayoutCubeWithRegions<T>(*this,HostMem);
  me->setRWStride(this->getRWStride());
  me->setRWPtrRegion(regId,rx,ry,z);
  limit = (limit)?(limit):(me->getStrideLimit());
  for(size_t i=0; i < limit; i++){
    T tmpvalue = me->read();
    char * Bptr = (char*) &tmpvalue;
    Bptr += (offsetBytes >0)?offsetBytes:0;
    value =  *((P*)Bptr);
    csvstring << value << delimiter;
  }
  if(this->getType() == DeviceGlobal)
    delete me;
  return csvstring.str();
}

template<typename T>
template<typename P>
string LayoutCubeWithRegions<T>::getCSVRegionPlane(size_t regId, size_t xlimit, size_t ylimit,  size_t plane, size_t offsetBytes, char delimiter)
{
  ostringstream csvstring;
  P value;
  LayoutCubeWithRegions<T> * me = this;
  if(this->getType() == DeviceGlobal)
     me = new LayoutCubeWithRegions<T>(*this,HostMem);
  me->setRWStrideX();
  xlimit = (xlimit>0)?( (xlimit > me->getRegW(regId))?(me->getRegW(regId)):(xlimit)):(me->getRegW(regId));
  ylimit = (ylimit>0)?( (ylimit > me->getRegH(regId))?(me->getRegH(regId)):(ylimit)):(me->getRegH(regId));
  for(size_t r=0; r < ylimit; r++)
  {
    me->setRWPtrRegion(regId,0,r,plane);
    csvstring << regId << delimiter << r << delimiter << '|' << delimiter;
    for(size_t c=0; c < xlimit; c++){
      T tmpvalue = me->read();
      char * Bptr = (char*) &tmpvalue;
      Bptr += (offsetBytes >0)?offsetBytes:0;
      value =  *((P*)Bptr);
      csvstring << value << delimiter;
    }
    csvstring << endl;
  }
  if(this->getType() == DeviceGlobal)
     delete me;
  return csvstring.str();
}


//creates a csv string for the data elements with limit (limit=0 == dim) data elements in z direction for each (x,y) data point within the region regId, data points are separated by new line
template<typename T>
template<typename P>
string LayoutCubeWithRegions<T>::getCSVRegionCube(size_t regId, size_t xlimit, size_t ylimit, size_t zlimit, size_t offsetBytes, char delimiter)
{
  ostringstream csvstring;
  P value;
  LayoutCubeWithRegions<T> * me = this;
  if(this->getType() == DeviceGlobal)
     me = new LayoutCubeWithRegions<T>(*this,HostMem);
  me->setRWStrideZ();
  xlimit = (xlimit>0)?( (xlimit>me->getRegW(regId))?(me->getRegW(regId) ):(xlimit)):(me->getRegW(regId));
  ylimit = (ylimit>0)?( (ylimit>me->getRegH(regId))?(me->getRegH(regId) ):(ylimit)):(me->getRegH(regId));
  zlimit = (zlimit>0)?( (zlimit>me->getDimZ())?(me->getDimZ()):(zlimit) ):(me->getDimZ());

  for(size_t r=0; r < ylimit ; r++)
  {
    for(size_t c=0; c < xlimit; c++){
      csvstring << regId << delimiter << c << delimiter << r << delimiter<< '|' << delimiter;
      me->setRWPtrRegion(regId,c,r,0);
      for(size_t z=0; z < zlimit; z ++){
        T tmpvalue = me->read();
        char * Bptr = (char*) &tmpvalue;
        Bptr += (offsetBytes >0)?offsetBytes:0;
        value =  *((P*)Bptr);
        csvstring << value << delimiter;
      }
      csvstring << endl;
    }
  }
  if(this->getType() == DeviceGlobal)
     delete me;
  return csvstring.str();
}




//prints a table of the elements at rx:ry:rz of every region casted to type of P which can be different from type T
//if elements are objects which do not offer a "<<" operator for printing a byte offset within the object to can be provided
//can be provided.
template<typename T>
template<typename P>
void LayoutCubeWithRegions<T>::printRegionTable( size_t rx, size_t ry, size_t rz, size_t offsetBytes)
{
  //only works for one value per region
  //assert(getRegW(0) == 1 && getRegW(0) == 1);
  ImgRegParams iP = getParams();
  size_t oldStride = this->getRWStride();
  this->setRWStride(iP.getRegW()); // iterate over regions
  size_t regRow = 0;
  for(regRow = 0; regRow < iP.getGridDimY(); regRow++){
    if(regRow == 0){
      //print index row
      cout << setw(5) << " " << "  ";
      for(size_t regCol = 0; regCol < iP.getGridDimX(); regCol++)
        cout<< setw(12) << regCol;
      cout << endl;
    }
    //print index column

    cout << setw(5) << regRow << ": ";
    this->setRWPtrRegion( iP.getGridDimX()*regRow,rx,ry,rz); //set RW pointer to first element of first region in grid row regRow
    for(size_t regCol = 0; regCol < iP.getGridDimX(); regCol++){
      T tmpvalue = this->read();
      char * Bptr = (char*) &tmpvalue;
      Bptr += (offsetBytes >0)?offsetBytes:0;
      P value = *((P*)Bptr);
      cout<< setw(12) << value;
    }
    cout << endl;
  }
  this->setRWStride(oldStride);
}

template <typename T>
template<typename P>
float LayoutCubeWithRegions<T>::getAvgReg(size_t regId, size_t plane, size_t offsetBytes, LayoutCubeWithRegions<unsigned short> * mask, unsigned short maskValue)
{

  ImgRegParams iP = getParams();

  if(mask != NULL)
    assert( iP == mask->getParams());
  size_t oldStride = this->getRWStride();
  this->setRWStrideX();
  if(mask != NULL) mask->setRWStrideX();

  double sum = 0;
  size_t cnt = 0;

  for(size_t ry=0; ry < iP.getRegH(regId); ry++){
    this->setRWPtrRegion(regId,0,ry,plane);
    if(mask != NULL) mask->setRWPtrRegion(regId,0,ry);
    for(size_t rx=0; rx < iP.getRegW(regId); rx++){
      T tmpvalue = this->read();
      char * Bptr = (char*) &tmpvalue;
      Bptr += (offsetBytes >0)?offsetBytes:0;
      P value = *((P*)Bptr);
      if(mask != NULL){
        unsigned short mv = mask->read();
        if(!(mv & maskValue)) continue;
      }
      sum += value;
      cnt ++;
    }
  }
  float ret = 0.0f;
  if(cnt > 0) ret = (float)(sum/cnt);
  this->setRWStride(oldStride);
  return ret;
}



template<typename T>
template<typename P>
float LayoutCubeWithRegions<T>::getStdDevReg(size_t regId, size_t plane, float avg, size_t offsetBytes, LayoutCubeWithRegions<unsigned short> * mask, unsigned short maskValue)
{


  ImgRegParams iP = getParams();

  if(mask != NULL)
    assert( mask->getParams() == iP);
  size_t oldStride = this->getRWStride();
  this->setRWStrideX();
  if(mask != NULL) mask->setRWStrideX();

  double sum = 0;
  size_t cnt = 0;

  for(size_t ry=0; ry < iP.getRegH(regId); ry++){
    setRWPtrRegion(regId,0,ry,plane);
    if(mask != NULL) mask->setRWPtrRegion(regId,0,ry);
    for(size_t rx=0; rx < iP.getRegW(regId); rx++){
      T tmpvalue = this->read();
      char * Bptr = (char*) &tmpvalue;
      Bptr += (offsetBytes >0)?offsetBytes:0;
      P value = *((P*)Bptr);
      if(mask != NULL){
        unsigned short mv = mask->read();
        if(!(mv & maskValue)) continue;
      }
      sum += (value - avg)*(value-avg);
      cnt ++;
    }
  }

  float ret = 0.0f;
  if(cnt > 0) ret = (float)sqrt(sum/cnt);
  this->setRWStride(oldStride);
  return ret;
}

//most parameters as above for CSV writer
// P has to provide << and == or a compare(P,P) function pointer
//limit will limit the comparison to #limit base elements and
//num  > 1 will compare num elements of type P starting from offsetBytes from a T element at regId,x,y,z
//if a mask is provided it will only be applied to the base coordinates designated by the stride of an entry.
//the mask will not be evaluated for elements designated by offset or num entries from the base coordinate
/*template <typename P>
  bool compare( LayoutCubeWithRegions<T> & that,
                size_t regId, size_t x=0, size_t y=0, size_t z=0,
                size_t limit = 0, size_t offsetBytes = 0, size_t num = 1,
                CCompare<P> *comparePtr = NULL,
                LayoutCubeWithRegions<unsigned short> * mask = NULL,
                unsigned short maskValueToMatch = 0,
                CompareOutput print= CompOutAll)*/

template<typename T>
template<typename P>
bool LayoutCubeWithRegions<T>::compare( LayoutCubeWithRegions<T> & that,
    size_t regId, size_t rx, size_t ry, size_t z,
    size_t limit, size_t offsetBytes, size_t num,
    CCompare<P> *comparePtr,
    LayoutCubeWithRegions<unsigned short> * mask,
    unsigned short maskValueToMatch,
    CompareOutput  verbose)
    {
  if(this->getSize() != that.getSize()){
    cout << "Compare: the two buffers are of different size" <<endl;
    return false;
  }
  if(mask != NULL){
    if(this->getDimX() != mask->getDimX() && this->getDimY() != mask->getDimY()){
      cout << "Compare: provided Map is of different size" <<endl;
      cout << "Data: ";
      this->getParams().print();
      cout << "Mask: ";
      mask->getParams().print();
      return false;

    }
  }
  unsigned short maskValue = 0;
  ImgRegParams iP;
  iP = getParams();
  LayoutCubeWithRegions<size_t>  RegionMap(iP.getGridDimX(), iP.getGridDimY());
  RegionMap.memSet(0);

  LayoutCubeWithRegions<T> * me = this;
  LayoutCubeWithRegions<T> * you = &that;

  //if Buffer is not in host memory make local copy
  if(this->getType() == DeviceGlobal)
    me = new LayoutCubeWithRegions<T>(*this,HostMem);
  if(that.getType() == DeviceGlobal)
    you = new LayoutCubeWithRegions<T>(that,HostMem);

  limit = (limit)?(limit):(this->getStrideLimit());
  size_t stride = this->getRWStride();

  me->setRWStride(stride);
  you->setRWStride(stride);
  me->setRWPtrRegion(regId,rx,ry,z);
  you->setRWPtrRegion(regId,rx,ry,z);
  size_t errorcnt = 0;

  for(size_t i = 0; i < limit; i++)
  {
    RegionMap.setRWPtrRegion(me->getRWlocRegId());
    if(mask!=NULL) maskValue = (*mask)[me->getRWlocXYIdx()];
    char * thisptr =(char*) &me->ref();
    char * thatptr =(char*) &you->ref();

    if(maskValue == maskValueToMatch || (maskValue & maskValueToMatch) ){ //also cover 0 case by using ==

      thisptr += (offsetBytes >0)?offsetBytes:0;
      thatptr += (offsetBytes >0)?offsetBytes:0;
      P *thisPptr = (P*)thisptr;
      P *thatPptr = (P*)thatptr;
      size_t numerr = 0;
      for(size_t j = 0; j< num; j++){
        P thisPvalue = thisPptr[j];
        P thatPvalue = thatPptr[j];

        if(comparePtr == NULL){
          if(thisPvalue != thatPvalue){
            if(verbose & CompOutAll) cout << this->getRWlocX() << ", " <<  this->getRWlocY() << ", ";
            if((verbose & CompOutAll) && num > 1) cout << j << ", ";
            if(verbose & CompOutAll) cout << thisPvalue<< ", "  << thatPvalue;
            errorcnt++; numerr++;
            if(verbose & CompOutAll) cout << ", " << abs(thisPvalue - thatPvalue) << endl;
            (RegionMap.ref())++;
          }
        }else{
          if(!comparePtr->Compare(thisPvalue,thatPvalue)){
            if(verbose & CompOutAll) cout << this->getRWlocX() << ", " <<  this->getRWlocY() << ", ";
            if((verbose & CompOutAll) && num > 1) cout << j << ", ";
            if(verbose & CompOutAll) cout << thisPvalue<< ", "  << thatPvalue;
            errorcnt++; numerr++;
            if(verbose & CompOutAll){
              cout << ", ";
              comparePtr->printDifference();
              cout << endl;
            }
            (RegionMap.ref())++;
          }
        }
        if( verbose != CompOutNone && num > 1 && j == num-1){
          if(numerr > 0)
            cout <<  "Compare: " << numerr << " out of " << num << " entries in element " << i << " are not equal!" << endl;
          else
            cout <<  "Compare: all " << num << " entries in element " << i << " are equal!" << endl;
        }
      }
    }
  }
  if(errorcnt>0){
    cout << "Compare: " << errorcnt << " out of " << limit*num << " are not equal, test FAILD *******!" << endl;
    if(verbose & CompOutByRegion){
      cout << "errors by region:" << endl;
      RegionMap.printRegionTable<size_t>();
    }
  }else
    cout << "Compare: all " << limit*num << " values are equal, test PASSED!" << endl;
  if(this->getType() == DeviceGlobal)
    delete me;
  if(that.getType() == DeviceGlobal)
    delete you;

  return (errorcnt>0)?(false):(true);
    }


template<typename T>
bool LayoutCubeWithRegions<T>::checkCoords(size_t regIdx, size_t rx, size_t ry, size_t z) const
{
  //call overloaded parent function
  if( LayoutCube<T>::checkCoords(regBaseX(regIdx)+rx, regBaseY(regIdx)+ry ,z)) return true;
  cerr << "provided coordinates  regId: " << regIdx << " (" << rx << "," << ry << "," << z << ")  are out of bounds and will cause segfault if accessed" << endl;
  assert(LayoutCube<T>::checkCoords(regBaseX(regIdx)+rx, regBaseY(regIdx)+ry ,z));
  return false;
}




#endif /* LAYOUTCUBEREGIONSTEMPLATE_H_ */
