/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RAWSPATIAL_H
#define RAWSPATIAL_H

#include <QObject>
#include "SpatialPlot.h"
#include "qcustomplot.h"

class RawSpatial : public SpatialPlot
{
public:
    RawSpatial(QWidget *parent);
    void doConvert(int &loading);
    void SetOption(QString option, int state);

protected:
    virtual float Get_Data(int frame, int y, int x);


private:
    void NeighborSubtract(short int *raw, int h, int w, int npts, uint16_t *mask, int ref);
    void GainCorrect(short int *raw, int h, int w, int npts);
    void TakeStdDev();

    int zeroState=0;
    int RowNoiseState=0;
    int ColNoiseState=0;
    int gainCorrState=0;
    int neighborState=0;
    int RefState=0;
    int EmptySubState=0;
    int ColFlState=0;
    int AdvcState=0;
    int stdState=0;
    int MaskReads=0;

    // thread copy of app specific stuff

    // thread specific stuff
    short *out=NULL;
    int *timestamps=NULL;
    int uncompFrames=0;
    int imageState=0;
    short *gainImage=NULL;


    int rowNoiseRemoved=0;
    int colNoiseRemoved=0;
    int gainCorrApplied=0;
    int neighborSubApplied=0;
    int RefSubApplied=0;
    int EmptySubApplied=0;
    int ColFlApplied=0;
    int AdvcApplied=0;
    int stdApplied=0;

};

#endif // RAWSPATIAL_H
