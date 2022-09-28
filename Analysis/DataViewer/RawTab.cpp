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
    {"noPCA",10,Qt::Unchecked},
    {"T0Correct",11,Qt::Unchecked},
    {"Navigate",12,Qt::Unchecked},
	{"Histogram",13,Qt::Unchecked},
	{"AverageSub",14,Qt::Unchecked},
	{"BitsNeeded",15,Qt::Unchecked},
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
    //m_button = new QPushButton("My Button", this);
        // set size and location of the button
    //m_button->setGeometry(QRect(QPoint(100, 100),QSize(100, 50)));

    // Connect button signal to appropriate slot
    //connect(m_button, SIGNAL (released()), this, SLOT (handleButton()));

    connect(mSpatialPlot, SIGNAL (fileNameChanged(QString)), this, SLOT (fileNameChangedRx(QString)));
}

void RawTab::fileNameChangedRx(QString fname)
{
	qDebug() << __PRETTY_FUNCTION__ << ": recevied/emitting fileNameChanged";
	emit fileNameChanged(fname);
}

