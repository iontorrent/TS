/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BFMASKSPATIAL_H
#define BFMASKSPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"
#include "modeltab.h"


class BfMaskSpatial : public SpatialPlot
{
public:
    BfMaskSpatial(QWidget *parent);
    void doConvert(int &loading);
    void addMask(uint mask);
    void RemoveMask(uint mask);
    void SetOption(QString txt, int state);
    void SetOptions(const maskCheckBox_t *_mcbt, int _mNitems);

protected:
    virtual float Get_Data(int idx, int y, int x);

private:
    void SetMask(int aligned, int state);
    uint16_t *out=NULL;
    uint mMaskVal=0;
    const maskCheckBox_t *mcbt=NULL;
    int mNitems=-1;
};

#endif // BFMASKSPATIAL_H
