/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "WellsSpatial.h"
#include "RawWells.h"
#include <iostream>
#include <string>

using namespace std;

WellsSpatial::WellsSpatial(QWidget *parent): SpatialPlot(parent)
{
}

void WellsSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    if(txt == "Show Masked")
        MaskReads = state;

    render();
}

void WellsSpatial::doConvert(int &loading)
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
        fflush(stdout);
        last_fname = _fname;
        if(out)
            free(out);
        out=NULL;

        QByteArray ba = fname.toLatin1();
        char * fn = ba.data();

        RawWells wells(fn,0,0);
        //wells.SetSubsetToLoad(&col[0], &row[0], col.size());
        wells.OpenForRead();
        flows = wells.NumFlows();
        loading=2;
// LIMIT Flows
        if(flows > 200)
            flows = 200;
// LIMIT Flows
        rows = wells.NumRows();
        cols = wells.NumCols();
        out = (float *)malloc(sizeof(float)*rows*cols*flows);

        cout << "#cols=" << cols << endl;
        cout << "#rows=" << rows << endl;
        cout << "#nFlow=" << flows << endl;
        string flowOrder = wells.FlowOrder();
        cout << "#flowOrder=" << flowOrder << endl;
        for(int row=0; row<rows; row++) {
            for(int col=0; col<cols; col++) {
                const WellData *w = wells.ReadXY(col,row);
                for(int flow=0; flow<flows; flow++) {
                    out[flow*rows*cols + row*cols + col] = w->flowValues[flow];
                }
            }
         }

        traces_len=flows;
        for(int i=0;i<flows;i++)
            traces_ts[i]=i;
        loading=0;
        printf("\ndone\n\n");
    }
}

float WellsSpatial::Get_Data(int flow, int y, int x)
{
    float rc = 0;
    if(out){
        rc = out[flow*rows*cols + y*cols + x];
        if(MaskReads && mMask && mMask[y*cols+x])
            rc = 0; // only display non-masked items
    }
    return rc;
}

void WellsSpatial::DoubleClick_traces(int x, int y, int ts)
{
    qDebug() << __PRETTY_FUNCTION__ << ": Y" << QString::number(y) << "_X" << QString::number(x) << " TS " << ts;
}



