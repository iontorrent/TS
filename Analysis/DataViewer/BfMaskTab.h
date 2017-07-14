/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BFMASKTAB_H
#define BFMASKTAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "modeltab.h"
#include "BfMaskSpatial.h"

class Dialog;

class BfMaskTab : public ModelTab
{
    Q_OBJECT
public:
    explicit BfMaskTab(QString mName, Dialog *_mParent, QWidget *parent = 0);

signals:

public slots:

private:
    BfMaskSpatial *mBfSpatialPlot;
};

#endif // BFMASKTAB_H
