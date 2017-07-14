/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MODELTAB_H
#define MODELTAB_H

#include <QtWidgets>
#include <QWidget>
#include "qcustomplot.h"
#include "SpatialPlot.h"
//#include "dialog.h"

typedef struct {
    const char *name;
    int option;
    Qt::CheckState state;
}maskCheckBox_t;

class Dialog;

class ModelTab : public QWidget
{
    Q_OBJECT
public:
    explicit ModelTab(QString _mName, Dialog *_mParent, QWidget *parent = 0);
    void setfilename(QString filename);
    void Save(QTextStream &strm){strm << mName << endl; mSpatialPlot->Save(strm);}

signals:

public slots:
    void SliderChanged(int k);
    void mTraceCombobox_currentIndexChanged(const QString &txt);
    void slot_changed(const QModelIndex& topLeft, const QModelIndex&bootmRight);

protected:
    QTableView *MakeList(const maskCheckBox_t *mcbt, int nitems);
    QComboBox *MakeTraceCB();
    QGroupBox *MakeSlider(QString name);
    QCustomPlot *MakeTracePlot(int flow_based=0);

    QString mName="";
    QComboBox* combo;

    Dialog *mParent;
    QSlider *mSlider;
    SpatialPlot *mSpatialPlot;
    QCustomPlot *mtracePlot;
    QStandardItem* item[25]={NULL};
    QComboBox *mTraceComboBox;

};

#endif // MODELTAB_H
