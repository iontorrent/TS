/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NUMPYTAB_H
#define NUMPYTAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "NumpySpatial.h"
#include "modeltab.h"

class Dialog;

class NumpyTab : public ModelTab
{
    Q_OBJECT
public:
    explicit NumpyTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);

public slots:

private:
    NumpySpatial *mNumpySpatialPlot=NULL;
};

#endif // NUMPYTAB_H
