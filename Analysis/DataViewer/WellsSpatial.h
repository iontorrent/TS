/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLSSPATIAL_H
#define WELLSSPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class WellsRdrThread;

class WellsSpatial : public SpatialPlot
{
public:
    WellsSpatial(QWidget *parent);
    void doConvert(int &loading);
    void SetOption(QString option, int state);
    void CustomTracePlotAdder(double &xmin, double &xmax, double &ymin, double &ymax);

protected:
    float Get_Data(int flow, int y, int x);
    void DoubleClick_traces(int x, int y, int ts, float val);
    void UpdateTraceData();

private:
    WellsRdrThread *threads=NULL;

    int MaskReads=0;
    int doNormalization=0;
//    float *out=NULL;
};


class WellsRdrThread : public QThread
{
//    Q_OBJECT

public:
    WellsRdrThread(QObject *parent = 0);
    ~WellsRdrThread();
    void Abort();
//    void NormalizeWells(int w, int h, int startX, int startY, int32_t *out, char **outSeq, char **outAlignSeq, char *_fileName, int *_loading, int adder=0);
//    void SetParent(SpatialPlot *_parent);
    void LoadFile(char *fn, float **_wellsOut, float **_wellsNormOut, int *flows, char *flowOrder, int *flowOrderLen, int *_loading);


signals:
//    void renderedImage(const QImage &image);

protected:
    void run() Q_DECL_OVERRIDE;

private:

//    int w;
//    int h;
//    int startX;
//    int startY;
//    int32_t *out;
//    char   **outSeq;
//    char   **outAlignSeq;
    int *loading=NULL;
    float **wellsOut=NULL;
    float **wellsNormOut=NULL;
    int *flows;
    char *flowOrder;
    int *flowOrderLen;
    char WellsFileName[4096];
//    char barcodeHeader[256];
//    std::vector<int> *seqIndx;
//    std::vector<BamTools::BamAlignment> *alignments;


    QMutex mutex;
    QWaitCondition condition;
    bool abort=0;
};


#endif // WELLSSPATIAL_H
