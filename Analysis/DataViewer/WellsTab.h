/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLSTAB_H
#define WELLSTAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "modeltab.h"
#include "WellsSpatial.h"
#include "AlignmentSpatial.h"

class Dialog;

class WellsTab : public ModelTab
{
    Q_OBJECT
public:
    explicit WellsTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);

signals:

public slots:

private:

};

#endif // WELLSTAB_H
