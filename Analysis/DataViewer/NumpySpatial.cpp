/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "NumpySpatial.h"
#include "RawWells.h"
#include <iostream>
#include <string>

using namespace std;

NumpySpatial::NumpySpatial(QWidget *parent): SpatialPlot(parent)
{
}

void NumpySpatial::addMask(uint mask)
{
    mMask |= mask;
    render();
}

void NumpySpatial::RemoveMask(uint mask)
{
    mMask &= ~mask;
    render();
}

void NumpySpatial::doConvert(int &loading)
{
    if(fname == ""){
        if(out)free(out);
        out=NumpyOut=NULL;
        last_fname = _fname;
        traces_len=0;
    }
    if (last_fname != _fname){
        loading=1;
        // load a new dat file
        printf("triggered %s %s\n",last_fname.toLatin1().data(),_fname.toLatin1().data());
        last_fname = _fname;
        if(out)
            free(out);
        out=NumpyOut=NULL;

        QByteArray ba = fname.toLatin1();
        char * fn = ba.data();

        {
          //Note, fillMask is optional argument w/ default of true.
          FILE *in = fopen ( fn, "rb" );
          if(in){
              char magic[6]={0};
              unsigned char major_version=0;
              unsigned char minor_version=0;
              unsigned short header_len=0;
              int elements_read=0;
              int elem_size=0;
              int lrows=0,lcols=0;
              static const char *elem_size_identifier="descr\': \'|b";
              static const char *shape_size_identifier="shape\': (";

              elements_read = fread ( &magic, sizeof ( magic ), 1, in );
              if( (elements_read == 1) && (!memcmp(&magic[1],"NUMPY",5))){
                  elements_read = fread ( &major_version, sizeof ( major_version ), 1, in );
                  if( elements_read == 1 ){
                      elements_read = fread ( &minor_version, sizeof ( minor_version ), 1, in );
                      elements_read = fread ( &header_len, sizeof ( header_len ), 1, in );
                      if( elements_read == 1 ){
                          char *hdr = (char *)malloc(header_len+1);
                          hdr[header_len]=0; // make sure it's null terminated
                          elements_read = fread ( hdr, header_len, 1, in );
                          if( elements_read == 1 ){
                              // we now have the text descriptor of the table..
                              // parse out the element size, and dimensions
                              char *lenptr = strstr(hdr,elem_size_identifier);
                              if(lenptr){
                                  lenptr += strlen(elem_size_identifier);
                                  sscanf(lenptr,"%d",&elem_size);
                                  printf("found element size of %d\n",elem_size);
                              }

                              char *shapeptr = strstr(hdr,shape_size_identifier);
                              if(shapeptr){
                                  shapeptr += strlen(shape_size_identifier);
                                  sscanf(shapeptr,"%d, %d)",&lrows,&lcols);
                                  printf("found %d rows and %d columns\n",lrows,lcols);
                              }

                              if(lrows && lcols && elem_size){


                                  int nbytes = sizeof(NumpyOut[0])*lrows*lcols;
                                  out = NumpyOut = (float *)malloc(nbytes);

                                  if(elem_size == 1){
                                      unsigned char *tmpStorage = (unsigned char *)malloc(lrows*lcols*1);

                                      elements_read = fread ( tmpStorage, 1, lrows*lcols, in );
                                      if( elements_read == lrows*lcols ){
                                          // success..
                                          for(int r=0;r<rows && r<lrows ;r++){
                                              for(int c=0;c<cols && c < lcols;c++){
                                                  out[r*cols+c] = tmpStorage[r*lcols+c];
                                              }
                                          }
                                      }
                                      free(tmpStorage);
                                  }
                              }
                          }
                      }
                  }
              }
              fclose ( in );
          }
        }
        traces_len=2;
        for(int i=0;i<traces_len;i++)
            traces_ts[i]=i;

        printf("\ndone loading Numpy File %s\n\n",fn);
        loading=0;
    }
}

float NumpySpatial::Get_Data(int frame, int y, int x)
{
    (void)frame;
    float rc = 0;
    if(out){
        rc = (out[y*cols + x]);//*1000);
    }
    return rc;
}



