/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "WellsSpatial.h"
#include "RawWells.h"
#include "../BaseCaller/WellsNormalization.h"
#include "../BaseCaller/BaseCallerParameters.h"
#include <iostream>
#include <string>

using namespace std;

OrderedDatasetWriter::OrderedDatasetWriter(){};
OrderedDatasetWriter::~OrderedDatasetWriter(){};
PerBaseQual::PerBaseQual(){};
PerBaseQual::~PerBaseQual(){};
ReadFilteringStats::ReadFilteringStats(){};
PhaseEstimator::PhaseEstimator(){};
void BaseCallerFilters::PrintHelp(){};
void PhaseEstimator::PrintHelp(){};
void PerBaseQual::PrintHelp(){};
void BarcodeClassifier::PrintHelp(){};
void MolecularTagTrimmer::PrintHelp(bool){};
void BaseCallerMetricSaver::PrintHelp(){};
void SaveJson(Json::Value const&, std::string const&){};


WellsSpatial::WellsSpatial(QWidget *parent): SpatialPlot(parent)
{
    Spa_Upper_Override=5.0f;
    Spa_Lower_Override=0.0f;
}

void WellsSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    if(txt == "Show Masked")
        MaskReads = state;
    if(txt == "Normalize Reads")
        doNormalization = state;

    render();
}

void WellsSpatial::doConvert(int &loading)
{

    if(fname == ""){
        free(WellsOut);
        WellsOut=NULL;
        last_fname = _fname;
        traces_len=0;
    }
    if (last_fname != _fname){
        // load a new dat file
        printf("triggered %s %s\n",last_fname.toLatin1().data(),_fname.toLatin1().data());
        fflush(stdout);
        last_fname = _fname;
        if(WellsOut)
            free(WellsOut);
        WellsOut=NULL;

        QByteArray ba = fname.toLatin1();
        char * fn = ba.data();

        if(threads == NULL){
            threads=new WellsRdrThread();
        }
        if(!loading)
            threads->LoadFile(fn,&WellsOut,&WellsNormOut,&flows,&flowOrder[0],&flowOrderLen,&loading);

        printf("\ndone\n\n");
    }
}




void WellsSpatial::CustomTracePlotAdder(double &xmin, double &xmax, double &ymin, double &ymax)
{
    (void)xmin;
    (void)xmax;
    if(ymin > -0.5)
        ymin=-0.5;
    if(ymax < 5)
        ymax=5;
}

float WellsSpatial::Get_Data(int flow, int y, int x)
{
    float rc = 0;
    if(doNormalization && WellsNormOut){
        rc = WellsNormOut[flow*rows*cols + y*cols + x];
        if(MaskReads && mMask && mMask[y*cols+x])
            rc = 0; // only display non-masked items
    }else if(WellsOut){
        rc = WellsOut[flow*rows*cols + y*cols + x];
        if(MaskReads && mMask && mMask[y*cols+x])
            rc = 0; // only display non-masked items
    }
    return rc;
}

void WellsSpatial::UpdateTraceData()
{
    traces_len=flows;
    for(int i=0;i<traces_len;i++)
        traces_ts[i]=i;
    SpatialPlot::UpdateTraceData();
}

void WellsSpatial::DoubleClick_traces(int x, int y, int ts, float val)
{
    qDebug() << __PRETTY_FUNCTION__ << ": Y" << QString::number(y) << "_X" << QString::number(x) << " TS " << ts << "val " << QString::number(val);
    curDisplayFlow = ts; // switch raw display to this flow
    GblTracesVer++; // force it to call render

}


void WellsRdrThread::LoadFile(char *fn, float **_wellsOut, float **_wellsNormOut, int *_flows, char *_flowOrder, int *_flowOrderLen, int *_loading)
{
    strcpy(WellsFileName,fn);
    wellsOut=_wellsOut;
    wellsNormOut = _wellsNormOut;
    printf("%s: %p %p\n",__FUNCTION__,wellsOut,wellsNormOut);
    flows=_flows;
    flowOrder=_flowOrder;
    flowOrderLen=_flowOrderLen;
    loading=_loading;

    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        __sync_add_and_fetch(loading,1);
        start(LowPriority);
    } else {
        //restart = true;
        condition.wakeOne();
    }
}

WellsRdrThread::WellsRdrThread(QObject *parent)
    : QThread(parent)
{
}

//! [0]

//! [1]
WellsRdrThread::~WellsRdrThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void WellsRdrThread::Abort()
{
    abort=true;
    mutex.unlock();
}

void WellsRdrThread::run()
{
    if(abort)
        return; // application is closed
    // normalize the wells data into wells
    RawWells wells(WellsFileName,0,0);
    //wells.SetSubsetToLoad(&col[0], &row[0], col.size());
    wells.OpenForRead();
    *flows = wells.NumFlows();
    *loading=2;

// LIMIT Flows
//    if(flows > 200)
//        flows = 200;
// LIMIT Flows
    int rows = wells.NumRows();
    int cols = wells.NumCols();
    (*wellsOut) = (float *)malloc(sizeof(float)*rows*cols*(*flows));
    (*wellsNormOut) = (float *)malloc(sizeof(float)*rows*cols*(*flows));

    cout << "#cols=" << cols << endl;
    cout << "#rows=" << rows << endl;
    cout << "#nFlow=" << *flows << endl;
    string _flowOrder = wells.FlowOrder();
    cout << "#flowOrder=" << _flowOrder << endl;
    strncpy(flowOrder,_flowOrder.c_str(),4096);
    flowOrder[4096-1]=0; // null terminate
    *flowOrderLen = strlen(flowOrder);

    {
        for(int row=0; row<rows; row++) {
            for(int col=0; col<cols; col++) {
                const WellData *w = wells.ReadXY(col,row);
                for(int flow=0; flow<(*flows); flow++) {
                    (*wellsOut)[flow*rows*cols + row*cols + col] = w->flowValues[flow];
                }
            }
         }

    }
        //if(doNormalization)
    {
//TODO: make mask_fn
        char mask_fn[2048];
        strcpy(mask_fn,WellsFileName);

        char *ptr = strstr(mask_fn,"1.wells");
        if(ptr)
            sprintf(ptr,"bfmask.bin");

        Mask mMask(mask_fn);
        //ion::FlowOrder flowOrd(wells.FlowOrder(),wells.NumFlows());

        OptArgs opts;
        BaseCallerContext bc;
        bc.SetKeyAndFlowOrder(opts, wells.FlowOrder(), wells.NumFlows());


        WellsNormalization wells_norm(&bc.flow_order, "default");
        wells_norm.SetWells(&wells, &mMask);
        wells.ReadWells();
        wells_norm.CorrectSignalBias(bc.keys);
        wells_norm.DoKeyNormalization(bc.keys);

        for(int row=0; row<rows; row++) {
            for(int col=0; col<cols; col++) {
                const WellData *w = wells.ReadXY(col,row);
                for(int flow=0; flow<(*flows); flow++) {
                    (*wellsNormOut)[flow*rows*cols + row*cols + col] = w->flowValues[flow];
                }
            }
         }
    }
    *loading=0; // done...
}



