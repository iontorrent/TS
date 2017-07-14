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

protected:
    float Get_Data(int frame, int y, int x);
    void ReadFile(char *fn);
    void UpdateTracePlot();

private:
    int32_t *out=NULL;
    AlignmentRdrThread *threads[100]={NULL};
    std::vector<int> seqIndx;
    std::vector<BamTools::BamAlignment> alignments;
    BamTools::BamAlignment *Get_Alignment(int y, int x);
    void Get_ReferenceSeq(BamTools::BamAlignment *al, std::string &qseq, std::string &tseq);
    void SetMask(int aligned, int state);
};


class AlignmentRdrThread : public QThread
{
//    Q_OBJECT

public:
    AlignmentRdrThread(QObject *parent = 0);
    ~AlignmentRdrThread();
    void Abort();
    void LoadFile(int w, int h, int startX, int startY, int32_t *out, char *_fileName, int *_loading, std::vector<int> *_seqIndx, std::vector<BamTools::BamAlignment> *_alignments);
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
    int *loading=NULL;
    char fileName[4096];
    std::vector<int> *seqIndx;
    std::vector<BamTools::BamAlignment> *alignments;


    QMutex mutex;
    QWaitCondition condition;
    bool abort=0;
};


#endif // ALIGNMENTSPATIAL_H
