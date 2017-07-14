/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "WellsTab.h"

static const maskCheckBox_t mcbt1[] = {
    {"Show Masked",8,Qt::Unchecked}

};

WellsTab::WellsTab(QString _mName, Dialog *_mParent, QWidget *parent)
    : ModelTab(_mName, _mParent, parent)
{
    mSpatialPlot = new WellsSpatial(this);

    QGridLayout *grid = new QGridLayout;
    grid->addWidget(MakeSlider("flows"),0,0);
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
