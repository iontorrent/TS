/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MICROSCOPETAB_H
#define MICROSCOPETAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "modeltab.h"
#include "MicroscopeSpatial.h"

class Dialog;

class MicroscopeTab : public ModelTab
{
    Q_OBJECT
public:
    explicit MicroscopeTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);
    void SetDualFileName(QString fn1, QString fn2);

signals:

public slots:

private:
    MicroscopeSpatial *mMicSpatial;

};

#endif // MICROSCOPETAB_H
