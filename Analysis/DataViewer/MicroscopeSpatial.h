/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MICROSCOPESPATIAL_H
#define MICROSCOPESPATIAL_H

#include <QObject>
#include <QThread>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class MicroscopeRdrThread;

#define MAX_MS_FN 10

class MicroscopeSpatial : public SpatialPlot
{
public:
    MicroscopeSpatial(QWidget *parent);
    void doConvert(int &loading);
    void SetOption(QString option, int state);
    void setSlider(int per);
    QImage *doRender();
    void SetDualFileName(QString fn1, QString fn2);

protected:
//    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;
    float Get_Data(int frame, int y, int x);
//    QLabel imageLabel;
    QImage QimgSrc[MAX_MS_FN];
//    QScrollArea scrollArea;

    uint GetPixVal(int y, int x);

    float pixelHeight[MAX_MS_FN]={-1};
    float pixelWidth[MAX_MS_FN]={-1};
    float pixelXStart[MAX_MS_FN]={-1};
    float pixelYStart[MAX_MS_FN]={-1};
    bool loaded[MAX_MS_FN] = {0};

    QString fileNameImg[MAX_MS_FN];
    QString fileNameCsv[MAX_MS_FN];


private:

};

#endif // MICROSCOPESPATIAL_H
