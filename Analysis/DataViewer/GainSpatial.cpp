/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "GainSpatial.h"
#include "RawWells.h"
#include <iostream>
#include <string>

using namespace std;

GainSpatial::GainSpatial(QWidget *parent): SpatialPlot(parent)
{
}

void GainSpatial::addMask(uint mask)
{
    mMask |= mask;
    render();
}

void GainSpatial::RemoveMask(uint mask)
{
    mMask &= ~mask;
    render();
}

void GainSpatial::doConvert(int &loading)
{
    if(fname == ""){
        free(out);
        out=gainVals=NULL;
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
        out=gainVals=NULL;

        QByteArray ba = fname.toLatin1();
        char * fn = ba.data();

        {
          //Note, fillMask is optional argument w/ default of true.
          FILE *in = fopen ( fn, "rb" );
          if(in){
              int magic=0, version=0, elements_read=0;
              int lrows=0,lcols=0;

              elements_read = fread ( &magic, sizeof ( magic ), 1, in );
              if( (elements_read == 1) && ((uint)magic == 0xFF115E3A)){
                  elements_read = fread ( &version, sizeof ( version ), 1, in );
                  if( elements_read == 1 ){
                      elements_read = fread ( &lrows, sizeof ( rows ), 1, in );
                      if( elements_read == 1 ){
                          elements_read = fread ( &lcols, sizeof ( cols ), 1, in );
                          if( elements_read == 1 ){
                              int nbytes = sizeof(out[0])*rows*cols;
                              out = gainVals = (float *)malloc(nbytes);
                              elements_read = fread ( out, nbytes, 1, in );
                              if( elements_read == 1 ){
                                  // success..
                                  if((cols ==0 || cols == lcols) &&
                                          (rows == 0 || rows == lrows)){
                                    cols = lcols;
                                    rows = lrows;
                                  }
                                  else{
                                      free(out);
                                      out=NULL;
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

        printf("\ndone loading Gain File %s\n\n",fn);
        loading=0;
    }
}

float GainSpatial::Get_Data(int frame, int y, int x)
{
    (void)frame;
    float rc = 0;
    if(out){
        rc = (out[y*cols + x]*1000);
    }
    return rc;
}



