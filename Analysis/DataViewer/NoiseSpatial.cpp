/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "NoiseSpatial.h"
#include "RawWells.h"
#include "dialog.h"
#include <iostream>
#include <string>

using namespace std;

NoiseSpatial::NoiseSpatial(QWidget *parent): SpatialPlot(parent)
{
}

void NoiseSpatial::addMask(uint mask)
{
    mMask |= mask;
    render();
}

void NoiseSpatial::RemoveMask(uint mask)
{
    mMask &= ~mask;
    render();
}

void NoiseSpatial::doConvert(int &loading)
{
    if(fname == ""){
        free(out);
        out=NULL;
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
        out=NULL;

        QByteArray ba = fname.toLatin1();
        char * fn = ba.data();

        {
          //Note, fillMask is optional argument w/ default of true.
          FILE *in = fopen ( fn, "rb" );
          if(in){
              uint magic=0, version=0, elements_read=0;
              uint lrows=0,lcols=0;
              uint RawExpmt_Y=Block_Y;
              uint RawExpmt_X=Block_X;

              elements_read = fread ( &magic, sizeof ( magic ), 1, in );
              if( (elements_read == 1) && (magic == 0xFF115E3A)){
                  elements_read = fread ( &version, sizeof ( version ), 1, in );
                  if( elements_read == 1 ){
                      elements_read = fread ( &lcols, sizeof ( lcols ), 1, in );
                      if( elements_read == 1 ){
                          elements_read = fread ( &lrows, sizeof ( lrows ), 1, in );
                          if( elements_read == 1 ){

                              // we're ready to read in the data...
                              int nbytes=sizeof(out[0])*rows*cols;
                              out = (uint16_t *)(uint16_t *)malloc(nbytes);

                              fseek(in, (RawExpmt_Y*lcols + RawExpmt_X)*sizeof(out[0]),SEEK_CUR);// seek to the right part of the row
                              for(int r=0;r<rows;r++){
                                  elements_read = fread ( &out[r*cols], cols*sizeof(out[0]), 1, in ); // seek to the right part of the row
                                  fseek(in, (lcols-cols)*sizeof(out[0]),SEEK_CUR);
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

        printf("\ndone loading Noise File %s\n\n",fn);
        loading=0;
    }
}

float NoiseSpatial::Get_Data(int idx, int y, int x)
{
    (void)idx;
    float rc = 0;
    if(out){
        rc = (out[y*cols + x]);
    }
    return rc;
}



