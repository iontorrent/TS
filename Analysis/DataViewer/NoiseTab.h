/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NOISETAB_H
#define NOISETAB_H

#include <QtWidgets>
#include <QWidget>
#include "NoiseSpatial.h"
#include "modeltab.h"

class Dialog;

class NoiseTab : public ModelTab
{
    Q_OBJECT
public:
    explicit NoiseTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);

signals:

public slots:

private:
};

#endif // NOISETAB_H
