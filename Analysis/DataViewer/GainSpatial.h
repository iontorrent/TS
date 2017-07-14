/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GAINSPATIAL_H
#define GAINSPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class GainSpatial : public SpatialPlot
{
public:
    GainSpatial(QWidget *parent);
    void doConvert(int &loading);
    void addMask(uint mask);
    void RemoveMask(uint mask);

protected:
    virtual float Get_Data(int idx, int y, int x);

private:
    float *out=NULL;
    uint mMask=0;
};

#endif // GAINSPATIAL_H
