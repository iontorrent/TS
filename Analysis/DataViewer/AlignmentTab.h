/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ALIGNMENTTAB_H
#define ALIGNMENTTAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "modeltab.h"
#include "AlignmentSpatial.h"

class Dialog;

class AlignmentTab : public ModelTab
{
    Q_OBJECT
public:
    explicit AlignmentTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);

signals:

public slots:

private:

};

#endif // ALIGNMENTTAB_H
