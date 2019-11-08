/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MicroscopeTab.h"

static const maskCheckBox_t mcbt1[] = {
//    {"Mask UnAligned",1,Qt::Unchecked},
//    {"Mask Aligned",2,Qt::Unchecked},
//    {"Show NoBarcode",3,Qt::Unchecked},
//    {"Show Sequence",4,Qt::Checked},
//    {"Show Query/Align",5,Qt::Checked},
//    {"InvertQuality",6,Qt::Unchecked},
};

MicroscopeTab::MicroscopeTab(QString _mName, Dialog *_mParent, QWidget *parent)
    : ModelTab(_mName, _mParent, parent)
{
    mMicSpatial = new MicroscopeSpatial(this);
    mSpatialPlot = mMicSpatial;

    QGridLayout *grid = new QGridLayout;
    grid->addWidget(MakeSlider("quality"),0,0);
    grid->addWidget(MakeTraceCB(),1,0);
    grid->addWidget(MakeList(mcbt1,sizeof(mcbt1)/sizeof(mcbt1[0])),0,1,2,1);
    grid->setRowStretch(0,0);
    grid->setRowStretch(1,0);

    grid->addWidget(mSpatialPlot,2,0,1,4);
    grid->addWidget(MakeTracePlot(1),3,0,1,4);

    // Set the graph styles to no line
    for(int i=0;i<MAX_NUM_TRACES;i++){
        mtracePlot->graph(i)->setLineStyle(QCPGraph::lsNone);
        mtracePlot->graph(i)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle,4));
    }

    setLayout(grid);
}

void MicroscopeTab::SetDualFileName(QString fn1, QString fn2)
{
    mMicSpatial->SetDualFileName(fn1,fn2);
}
