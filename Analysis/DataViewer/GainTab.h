/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GAINTAB_H
#define GAINTAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "GainSpatial.h"
#include "modeltab.h"

class Dialog;

class GainTab : public ModelTab
{
    Q_OBJECT
public:
    explicit GainTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);

public slots:

private:
    GainSpatial *mGainSpatialPlot=NULL;
};

#endif // GAINTAB_H
