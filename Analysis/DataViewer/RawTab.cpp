/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RawTab.h"
#include "qcustomplot.h"
#include "deInterlace.h"
#include "CorrNoiseCorrector.h"

static const maskCheckBox_t mcbt1[] = {
    {"Zero traces",0,Qt::Unchecked},
    {"Rmv Row Noise",1,Qt::Unchecked},
    {"Rmv Col Noise",2,Qt::Unchecked},
    {"Gain Correct",3,Qt::Unchecked},
    {"NeighborSubtract",4,Qt::Unchecked},
    {"RefSubtract",5,Qt::Unchecked},
//    {"EmptySubtract",6,Qt::Unchecked},
    {"Column Flicker",6,Qt::Unchecked},
    {"Adv Compression",7,Qt::Unchecked},
    {"Show Masked",8,Qt::Unchecked},
    {"StdDev",9,Qt::Unchecked},
};

RawTab::RawTab(QString _mName, Dialog *_mParent, QWidget *parent)
    : ModelTab(_mName, _mParent, parent)
{
    mSpatialPlot = new RawSpatial(this);

    QGridLayout *grid = new QGridLayout;
    grid->addWidget(MakeSlider("frames"),0,0);
    grid->addWidget(MakeTraceCB(),1,0);
    grid->addWidget(MakeList(mcbt1,sizeof(mcbt1)/sizeof(mcbt1[0])),0,1,2,1);
    grid->setRowStretch(0,0);
    grid->setRowStretch(1,0);
    grid->addWidget(mSpatialPlot,2,0,1,4);
    grid->addWidget(MakeTracePlot(),3,0,1,4);

    // add compression points to graph
    for(int i=0;i<MAX_NUM_TRACES;i++){
        mtracePlot->graph(i)->setScatterStyle(QCPScatterStyle::ssCircle);
    }

    setLayout(grid);
}


