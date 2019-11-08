/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NUMPYSPATIAL_H
#define NUMPYSPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class NumpySpatial : public SpatialPlot
{
public:
    NumpySpatial(QWidget *parent);
    void doConvert(int &loading);
    void addMask(uint mask);
    void RemoveMask(uint mask);

protected:
    virtual float Get_Data(int idx, int y, int x);

private:
    float *out=NULL;
    uint mMask=0;
};

#endif // NUMPYSPATIAL_H
