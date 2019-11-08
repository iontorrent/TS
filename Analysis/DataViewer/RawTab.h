/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RAWTAB_H
#define RAWTAB_H

#include <QtWidgets>
#include <QWidget>
#include "dialog.h"
#include "RawSpatial.h"
#include "qcustomplot.h"
#include "modeltab.h"

class Dialog;

class RawTab : public ModelTab
{
    Q_OBJECT

public:
    explicit RawTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);
    QPushButton *m_button;

signals:
	void fileNameChanged(QString fname);

public slots:
	void fileNameChangedRx(QString fname);
	//void handleButton();

};

#endif // RAWTAB_H
