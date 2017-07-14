/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLSSPATIAL_H
#define WELLSSPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class WellsSpatial : public SpatialPlot
{
public:
    WellsSpatial(QWidget *parent);
    void doConvert(int &loading);
    void SetOption(QString option, int state);

protected:
    float Get_Data(int flow, int y, int x);
    void DoubleClick_traces(int x, int y, int ts);

private:
    int MaskReads=0;
    float *out=NULL;
};

#endif // WELLSSPATIAL_H
