/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "modeltab.h"

ModelTab::ModelTab(QString _mName, Dialog *_mParent, QWidget *parent) : QWidget(parent)
{
    mParent = _mParent;
    mName=_mName;
}

QComboBox *ModelTab::MakeTraceCB()
{
    mTraceComboBox = new QComboBox;
    mTraceComboBox->setEditable(false);
    for(int i=0;i<MAX_NUM_TRACES;i++){
        mTraceComboBox->addItem(tr("Trace ") + QString::number(i));
    }
    mTraceComboBox->setFixedWidth(250);
//    mTraceComboBox->setFixedHeight(24);

    connect(mTraceComboBox, SIGNAL(currentIndexChanged(QString)),this,SLOT(mTraceCombobox_currentIndexChanged(QString)));
    return mTraceComboBox;
}

QTableView *ModelTab::MakeList(const maskCheckBox_t *mcbt, int nitems)
{
    QStandardItemModel *model = new QStandardItemModel(5, 3);
    int x=0;
    int y=0;
    for (int r = 0; r < nitems; ++r)
    {
        item[r] = new QStandardItem(mcbt[r].name);

        item[r]->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        item[r]->setData(mcbt[r].state, Qt::CheckStateRole);
        if(mcbt[r].state == Qt::Checked)
            mSpatialPlot->SetOption(mcbt[r].name,1);
        model->setItem(y++, x, item[r]);
        if(y > 4){
            x++;
            y=0;
        }
    }

    connect(model, SIGNAL(dataChanged ( const QModelIndex&, const QModelIndex&)), this, SLOT(slot_changed(const QModelIndex&, const QModelIndex&)));

    QTableView *list = new QTableView();
    list->setShowGrid(false);
    QHeaderView *verticalHeader = list->verticalHeader();
    verticalHeader->sectionResizeMode(QHeaderView::Fixed);
    verticalHeader->setDefaultSectionSize(15);
    verticalHeader->hide();
    list->horizontalHeader()->hide();

    list->setModel(model);
    list->horizontalHeader()->setSectionResizeMode(0,QHeaderView::Stretch);
    list->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
    list->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
//    list->setFixedWidth(350);
    list->setFixedHeight(78);
    return (list);
}



QCustomPlot *ModelTab::MakeTracePlot(int flow_based)
{
    mtracePlot = new QCustomPlot(this);
    mSpatialPlot->setTracePlot(mtracePlot,flow_based);
    return mtracePlot;
}


QGroupBox *ModelTab::MakeSlider(QString name)
{
    QGroupBox *frameBox = new QGroupBox(name);
    QVBoxLayout *vframeBox = new QVBoxLayout;

    mSlider = new QSlider(Qt::Horizontal);
    connect(mSlider, SIGNAL(valueChanged(int)), this, SLOT(SliderChanged(int)));
    vframeBox->addWidget(mSlider);
    mSlider->setFocusPolicy(Qt::NoFocus);
    frameBox->setLayout(vframeBox);
    frameBox->setMaximumHeight(50);
    frameBox->setMaximumWidth(250);
    if(name == ""){
        mSlider->hide();
    }
    return(frameBox);
}

void ModelTab::SliderChanged(int k)
{
    printf("%s: %d\n",__FUNCTION__,k);
    mSpatialPlot->setSlider(k);
}

void ModelTab::mTraceCombobox_currentIndexChanged(const QString &txt)
{
    printf("%s: %s\n",__FUNCTION__,txt.toLatin1().data());
    fflush(stdout);
    int selection=0;
    sscanf(txt.toLatin1().data(),"Trace %d",&selection);
    mSpatialPlot->setRawTrace(selection);
}

void ModelTab::setfilename(QString filename)
{
    printf("%s:\n",__FUNCTION__);
    mSpatialPlot->setfilename(filename);
}

void ModelTab::slot_changed(const QModelIndex& topLeft, const QModelIndex&bootmRight)
{
  (void)bootmRight;
  int index=topLeft.column()*5+topLeft.row();
  qDebug()  << __PRETTY_FUNCTION__ << ": Item " << index << " ";

//  if(index < mNitems)
  {
    bool state = (item[index]->checkState() == Qt::Checked);
    mSpatialPlot->SetOption(item[index]->text(),state);
  }
}
