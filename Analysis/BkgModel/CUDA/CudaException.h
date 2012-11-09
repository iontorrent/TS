/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAEXCEPTION_H 
#define CUDAEXCEPTION_H

#include <iostream>
#include <exception>



using namespace std;

class cudaException: public exception
{

  cudaError_t err;
  const char * file;
  int line;

public:

  cudaError_t getCudaError() { return err; };

  virtual const char* what() const throw()
  {
    return "CUDA: an Exception occured";
  }

    
  cudaException(cudaError_t err,  const char * file, int line):err(err),file(file),line(line)
  {};

  virtual void Print() const throw() 
  {
    
   cout << " +----------------------------------------" << endl
        << " | ** CUDA ERROR! ** " << endl                               
        << " | Error: " << err << endl                             
        << " | Msg: " << cudaGetErrorString(err) << endl           
        << " | File: " << file << endl                         
        << " | Line: " << line << endl                         
        << " +----------------------------------------" << endl << flush;  
  }

}; 



class cudaStreamCreationError: public cudaException
{
  virtual const char* what() const throw()
  {
    //Print();
    return "CUDA: could not acquire stream from streamPool";
  }

  public:
  cudaStreamCreationError( const char * file, int line):cudaException(cudaErrorUnknown,file,line) {};

};


class cudaAllocationError: public cudaException
{
  virtual const char* what() const throw()
  {
    //Print();
    return "CUDA: could not allocate memory";
  }

  public:
  cudaAllocationError(cudaError_t err, const char * file, int line):cudaException(err,file,line) {};

};

class cudaNotEnoughMemForStream: public cudaException
{
  virtual const char* what() const throw()
  {
    //Print();
    return "CUDA: Not enough memory for context and at least one stream!";
  }

 public:
  cudaNotEnoughMemForStream( const char * file, int line):cudaException(cudaErrorStartupFailure,file,line) {};

};

#endif  // CUDAEXCEPTION_H

