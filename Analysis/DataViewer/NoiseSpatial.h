/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NOISESPATIAL_H
#define NOISESPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class Dialog;

class NoiseSpatial : public SpatialPlot
{
public:
    NoiseSpatial(QWidget *parent);

    void doConvert(int &loading);
    void addMask(uint mask);
    void RemoveMask(uint mask);
    float Get_Data(int idx, int y, int x);

private:
    uint16_t *out=NULL;
    uint mMask=0;
};

#endif // NOISESPATIAL_H
