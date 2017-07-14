/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "GainTab.h"
#include "Mask.h"


static const maskCheckBox_t mcbt1[] = {
};


GainTab::GainTab(QString _mName, Dialog *_mParent, QWidget *parent)
    : ModelTab(_mName, _mParent,parent)
{
    mGainSpatialPlot = new GainSpatial(this);
    mSpatialPlot = mGainSpatialPlot;

    QGridLayout *grid = new QGridLayout;
    grid->addWidget(MakeSlider(""),0,0);
    grid->addWidget(MakeTraceCB(),1,0);
    grid->addWidget(MakeList(mcbt1,sizeof(mcbt1)/sizeof(mcbt1[0])),0,1,2,1);
    grid->setRowStretch(0,0);
    grid->setRowStretch(1,0);
    grid->addWidget(mSpatialPlot,2,0,1,4);
    grid->addWidget(MakeTracePlot(),3,0,1,4);
    setLayout(grid);
}


