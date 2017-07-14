/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#ifndef SPATIALPLOT_H
#define SPATIALPLOT_H

#include <QObject>
#include <QPixmap>
#include <QWidget>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include "qcustomplot.h"


class SpatialPlot;

#define MAX_NUM_TRACES 12
#define MAX_TRACE_LEN 500

typedef struct{
    int y;
    int x;
}TraceDescr_t;


//! [0]
class SpatialPlotThread : public QThread
{
    Q_OBJECT

public:
    SpatialPlotThread(QObject *parent = 0);
    ~SpatialPlotThread();
    void Abort();
    void render();
    void SetParent(SpatialPlot *_parent);


signals:
    void renderedImage(const QImage &image);

protected:
    void run() Q_DECL_OVERRIDE;

private:

    QMutex mutex;
    QWaitCondition condition;
    bool restart=0;
    bool abort=0;
    SpatialPlot *mParent;


};

//! [0]
class SpatialPlot : public QWidget
{
    Q_OBJECT

public:
    SpatialPlot(QWidget *parent = 0);
    ~SpatialPlot();
    void doConvertInt();
    virtual void doConvert(int &loading);
    virtual void copyData();
    QImage *doRender();
    void  copyLastCoord();
    void setRawTrace(int selection);
    void setfilename(QString fileName);
    void setSlider(int per);
    void setTracePlot(QCustomPlot *TracePlot, int _flow_based=0);

    void keyPressEvent(QKeyEvent *event) Q_DECL_OVERRIDE;
    void Save(QTextStream &strm);
    virtual void SetOption(QString option, int state);

    static void SetBlockCoord(int x, int y){Block_Y=y;Block_X=x;}

protected:

    virtual void DoubleClick(int x,int y);
    virtual void DoubleClick_traces(int x, int y, int ts);

    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;
#ifndef QT_NO_WHEELEVENT
    void wheelEvent(QWheelEvent *event) Q_DECL_OVERRIDE;
#endif
//    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
//    void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void render();
    uint rgbFromWaveLength(double wave);
    enum { ColormapSize = 512 };
    uint colormap[ColormapSize];
    virtual float Get_Data(int idx, int y, int x);

    static TraceDescr_t traces[MAX_NUM_TRACES];
    static int traces_curSelection;
    static int GblTracesVer;

    int LocalTracesVer=0;
    float traces_Val[MAX_NUM_TRACES][MAX_TRACE_LEN];

    float traces_ts[MAX_TRACE_LEN]={0};
    int _mTracePlotSet[MAX_TRACE_LEN]={0};
    int traces_len=0;
    void UpdateTraceData();
    virtual void UpdateTracePlot();
    QCustomPlot *_mTracePlot=NULL;
    int flow_based=0;
    int ignore_pinned=0;


protected slots:
    virtual void updatePixmap(const QImage &image);
    void zoom(double zoomFactor, double XWeight=0.5, double YWeight=0.5);
    void showPointToolTip(QMouseEvent *event);
    void showPointToolTip_traces(QMouseEvent *event);

private:
    void mouseDoubleClickEvent(QMouseEvent *event);
    void mouseDoubleClickEvent_traces(QMouseEvent *event);

    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void scroll(int deltaX, int deltaY);
    int GetAY(int y, int Height, int & ooby);
    int GetAX(int x, int Width, int & oobx);

    int GetY(int ay, int Height, int & ooby);
    int GetX(int ax, int Width, int & oobx);

    SpatialPlotThread thread;
    QPixmap pixmap;
    QPoint pixmapOffset;
    QPoint lastDragPos;
    bool firstShow=false;

    // mouse location variables
    int LastPressX=0;
    int LastPressY=0;
    float _mTraceRange=0;


protected:
    // window variables
    static int startX;
    static int endX;
    static int startY;
    static int endY;
    static int frame;
    static int flow;

    static int Block_Y;
    static int Block_X;

    int lastStartX=0;
    int lastStartY=0;
    int lastEndX=0;
    int lastEndY=0;
    int initialHints=1;
    int loading=0;

    int maxPixVal=0;
    int minPixVal=0;

    QString fname="";

    // thread copy of app specific stuff
    QString _fname="";
    QString last_fname="";

    // image variables
    static int rows;
    static int cols;
    static int frames;
    static int flows;

    static const int borderHeight=20;
    static const int borderWidth=30;
    static const int scaleBarWidth=35;

    static uint16_t *refMask; //from BfMaskTab
    static float *gainVals;   //from GainTab
    static char *mMask; // only display pixels with a 0

};
//! [0]

#endif // SPATIALPLOT_H
