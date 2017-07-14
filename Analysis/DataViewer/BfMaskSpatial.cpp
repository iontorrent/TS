/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BfMaskSpatial.h"
#include "RawWells.h"
#include <iostream>
#include <string>

using namespace std;

BfMaskSpatial::BfMaskSpatial(QWidget *parent): SpatialPlot(parent)
{
}

void BfMaskSpatial::addMask(uint mask)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << mask;
    mMaskVal |= mask;
    render();
}

void BfMaskSpatial::RemoveMask(uint mask)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << mask;
    mMaskVal &= ~mask;
    render();
}

void BfMaskSpatial::SetMask(int inverted, int state)
{
    if(mMask){
        // set the global mask bits for aligned or unaligned reads
        if(!state){
            // clear the mask bits for all reads
            for(int i=0;i<(rows*cols);i++)
                mMask[i] &= ~0x2;
        }else{
            for(int i=0;i<(rows*cols);i++){
                if((inverted && ((out[i] & mMaskVal) == 0)) || (!inverted && (out[i] & mMaskVal)))
                    mMask[i] |= 0x2;
            }
        }
    }
}

void BfMaskSpatial::doConvert(int &loading)
{
    if(fname == ""){
        free(out);
        out=refMask=NULL;
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
        out=refMask=NULL;

        QByteArray ba = fname.toLatin1();
        char * fn = ba.data();

        {
          //Note, fillMask is optional argument w/ default of true.
          FILE *in = fopen ( fn, "rb" );
          if(in){
              int elements_read = fread ( &rows, sizeof ( rows ), 1, in );
              assert ( elements_read == 1 );
              elements_read = fread ( &cols, sizeof ( cols ), 1, in );
              assert ( elements_read == 1 );
              int nbytes = sizeof(out[0])*rows*cols;
              out = refMask = (uint16_t *)malloc(nbytes);
              elements_read = fread ( out, nbytes, 1, in );
              assert ( elements_read == 1 );
              fclose ( in );
          }
        }
        traces_len=2;
        for(int i=0;i<traces_len;i++)
            traces_ts[i]=i;
        printf("\ndone loading BfMask %s\n\n",fn);
        loading=0;
    }
}


float BfMaskSpatial::Get_Data(int idx, int y, int x)
{
    (void)idx;
    float rc = 0;
    if(out){
        rc = (out[y*cols + x] & mMaskVal);
    }
    return rc;
}

void BfMaskSpatial::SetOptions(const maskCheckBox_t *_mcbt, int _mNitems)
{
    mcbt=_mcbt;
    mNitems=_mNitems;
}

void BfMaskSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    if(txt == "ApplyMask")
        SetMask(0,state);
    else if(txt == "ApplyInvMask")
        SetMask(1,state);
    else{
        for(int idx=0;idx<mNitems;idx++){
            if(mcbt[idx].name == txt){
                if(state)
                    addMask(1<<mcbt[idx].option);
                else{
                    RemoveMask(1<<mcbt[idx].option);
                }
            }
        }
    }
    render();
}

