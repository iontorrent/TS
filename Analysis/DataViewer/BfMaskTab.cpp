/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BfMaskTab.h"
#include "Mask.h"



static const maskCheckBox_t mcbt1[] = {
    {"MaskEmpty",0,Qt::Checked},
    {"MaskBead",1,Qt::Checked},
    {"MaskLive",2,Qt::Checked},
    {"MaskDud",3,Qt::Checked},
    {"MaskReference",4,Qt::Checked},
    {"MaskTF",5,Qt::Checked},
    {"MaskLib",6,Qt::Checked},
    {"MaskPinned",7,Qt::Unchecked},
    {"MaskIgnore",8,Qt::Unchecked},
    {"MaskWashout",9,Qt::Unchecked},
    {"MaskExclude",10,Qt::Unchecked},
//    {"MaskKeypass",11,Qt::Unchecked},
//    {"MaskFilteredBadKey",12,Qt::Unchecked},
//    {"MaskFilteredShort",13,Qt::Unchecked},
//    {"MaskFilteredBadPPF",14,Qt::Unchecked},
//    {"MaskFilteredBadResidual",15,Qt::Unchecked}
    {"ApplyMask",16,Qt::Unchecked},
    {"ApplyInvMask",17,Qt::Unchecked},

};

BfMaskTab::BfMaskTab(QString mName, Dialog *_mParent, QWidget *parent)
    : ModelTab(mName,_mParent,parent)
{
    mBfSpatialPlot = new BfMaskSpatial(this);
    mBfSpatialPlot->SetOptions(mcbt1,sizeof(mcbt1)/sizeof(mcbt1[0]));
    mSpatialPlot = mBfSpatialPlot;

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



