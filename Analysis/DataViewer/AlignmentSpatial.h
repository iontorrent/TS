/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ALIGNMENTSPATIAL_H
#define ALIGNMENTSPATIAL_H

#include <QObject>
#include <QThread>
#include "SpatialPlot.h"
#include "qcustomplot.h"
#include "api/BamReader.h"

class AlignmentRdrThread;

class AlignmentSpatial : public SpatialPlot
{
public:
    AlignmentSpatial(QWidget *parent);
    void doConvert(int &loading);
    void SetOption(QString option, int state);
    void CustomTracePlotAdder(double &xmin, double &xmax, double &ymin, double &ymax);
    void setSlider(int per);

protected:
    float Get_Data(int frame, int y, int x);
    void ReadFile(char *fn, int adder);
    void UpdateTraceData();
    void DoubleClick_traces(int x, int y, int ts, float val);
    void SetTracesToolTip(int y, int x, float ts, float val);

private:
    char *ApplyFilters(int x, int y);

    AlignmentRdrThread *threads[100]={NULL};
    int ShowNoBarcodes=0;
    int ShowSequence=0;
    int showQueryBases=0;
    int InvertQuality=0;
    int QualityLimit=0;
 //   std::vector<int> seqIndx;
 //   std::vector<BamTools::BamAlignment> alignments;
 //   BamTools::BamAlignment *Get_Alignment(int y, int x);
 //   void Get_ReferenceSeq(BamTools::BamAlignment *al, std::string &qseq, std::string &tseq);
    void SetMask(int aligned, int state);
};


class AlignmentRdrThread : public QThread
{
//    Q_OBJECT

public:
    AlignmentRdrThread(QObject *parent = 0);
    ~AlignmentRdrThread();
    void Abort();
    void LoadFile(int w, int h, int startX, int startY, int32_t *out, char **outSeq, char **outAlignSeq,
                  uint16_t *_outMapQuality, uint32_t **_outErrSeq, char *outFlowOrder, int *outflowOrderLen,
                  char *_fileName, int *_loading, int adder=0);
//    void SetParent(SpatialPlot *_parent);


signals:
//    void renderedImage(const QImage &image);

protected:
    void run() Q_DECL_OVERRIDE;

private:

    int w;
    int h;
    int startX;
    int startY;
    int32_t *out;
    char   **outSeq;
    char    *outFlowOrder;
    int     *outFlowOrderLen;
    char   **outAlignSeq;
    uint32_t **outErrSeq;
    uint16_t *outMapQuality;
    int *loading=NULL;
    int adder=0;
    char fileName[4096];
    char barcodeHeader[256];
//    std::vector<int> *seqIndx;
//    std::vector<BamTools::BamAlignment> *alignments;


    QMutex mutex;
    QWaitCondition condition;
    bool abort=0;
};


#endif // ALIGNMENTSPATIAL_H
