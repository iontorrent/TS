/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */

#include <QtWidgets>
#include <QFileInfo>

#include "dialog.h"

//! [0]
Dialog::Dialog()
{
    QTabWidget *tabWidget = new QTabWidget;
    tabWidget->addTab((mRawTab = new RawTab("Dat",this)), tr("Raw"));
    tabWidget->addTab((mBfMaskTab = new BfMaskTab("BfMask",this)), tr("BfMask"));
    tabWidget->addTab((mNoiseTab = new NoiseTab("Noise",this)), tr("Noise"));
    tabWidget->addTab((mGainTab = new GainTab("Gain",this)), tr("Gain"));
    tabWidget->addTab((mWellsTab = new WellsTab("Wells",this)), tr("1.wells"));
    tabWidget->addTab((mAlignmentTab = new AlignmentTab("Alignment",this)), tr("Alignment"));

    connect(tabWidget,SIGNAL(currentChanged(int)), this, SLOT(currentChanged(int)));

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(tabWidget);

    setLayout(mainLayout);
    setCentralWidget(tabWidget);


    setWindowTitle(tr("Ion Dat Explorer"));
    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));

    QAction *dummyAct = new QAction(tr(""),this);
    fileMenu->addAction(dummyAct);

    OpenDatAct = new QAction(tr("&Open Dat file"), this);
    OpenDatAct->setStatusTip(tr("Open Dat file"));
    connect(OpenDatAct, SIGNAL(triggered()), this, SLOT(browseDatDir()));
    fileMenu->addAction(OpenDatAct);

    OpenBfMaskAct = new QAction(tr("&Open BfMask file"), this);
    OpenBfMaskAct->setStatusTip(tr("Open Dat file"));
    connect(OpenBfMaskAct, SIGNAL(triggered()), this, SLOT(browseBfMaskDir()));
    fileMenu->addAction(OpenBfMaskAct);

    OpenNoiseAct = new QAction(tr("&Open Noise file"), this);
    OpenNoiseAct->setStatusTip(tr("Open Noise file"));
    connect(OpenNoiseAct, SIGNAL(triggered()), this, SLOT(browseNoiseDir()));
    fileMenu->addAction(OpenNoiseAct);

    OpenGainAct = new QAction(tr("&Open Gain file"), this);
    OpenGainAct->setStatusTip(tr("Open Gain file"));
    connect(OpenGainAct, SIGNAL(triggered()), this, SLOT(browseGainDir()));
    fileMenu->addAction(OpenGainAct);

    OpenWellsAct = new QAction(tr("&Open Wells file"), this);
    OpenWellsAct->setStatusTip(tr("Open Wells file"));
    connect(OpenWellsAct, SIGNAL(triggered()), this, SLOT(browseWellsDir()));
    fileMenu->addAction(OpenWellsAct);

    OpenBamAct = new QAction(tr("&Open Bam file"), this);
    OpenBamAct->setStatusTip(tr("Open Bam file"));
    connect(OpenBamAct, SIGNAL(triggered()), this, SLOT(browseBamDir()));
    fileMenu->addAction(OpenBamAct);

    QAction *ClearAct = new QAction(tr("&Clear"), this);
    ClearAct->setStatusTip(tr("Clear"));
    connect(ClearAct, SIGNAL(triggered()), this, SLOT(Clear()));
    fileMenu->addAction(ClearAct);

    dummyAct = new QAction(tr(""),this);
    fileMenu->addAction(dummyAct);
    QAction *SaveAct = new QAction(tr("&SaveData"), this);
    SaveAct->setStatusTip(tr("SaveData"));
    connect(SaveAct, SIGNAL(triggered()), this, SLOT(Save()));
    fileMenu->addAction(SaveAct);
    dummyAct = new QAction(tr(""),this);
    fileMenu->addAction(dummyAct);

    QAction *aboutAct = new QAction(tr("&about"), this);
    aboutAct->setStatusTip(tr("Create a new file"));
    connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));
    fileMenu->addAction(aboutAct);

    resize(800, 800);

}
//! [0]

void Dialog::Clear()
{
    DatFileName="";
    BfMaskFileName="";
    NoiseFileName="";
    GainFileName="";
    WellsFileName="";
    RawExpmtDir = "";
    RawExpmt_X = -1;
    RawExpmt_Y = -1;
    SpatialPlot::SetBlockCoord(RawExpmt_X,RawExpmt_Y);
    AutoCompletePaths();

}


void Dialog::Save()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                               tr("Save Charts File"), "","*.csv");

    printf("%s: %s\n",__FUNCTION__,fileName.toUtf8().constData());
    if (!fileName.isEmpty()) {
        // save all the charts to this file..
        QFile file(fileName);
        if (file.open(QIODevice::WriteOnly)) {
//            std::cerr << "Cannot open file for writing: "
//                      << qPrintable(file.errorString()) << std::endl;
            QTextStream out(&file);

            mRawTab->Save(out);
            mBfMaskTab->Save(out);
            mWellsTab->Save(out);
            mNoiseTab->Save(out);
            mGainTab->Save(out);
            mAlignmentTab->Save(out);
            file.close();
        }
    }
}

void Dialog::about()
//! [13] //! [14]
{
   QMessageBox::about(this, tr("About Application"),
            tr("<b>Ion Data Viewer Application</b> \n"
               "Version 0.1"));
}

void Dialog::browseDatDir()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                               tr("Open Dat File"), DatFileName,"*.dat*");

    printf("%s: %s\n",__FUNCTION__,fileName.toLatin1().data());
    if (!fileName.isEmpty()) {
        DatFileName=fileName;
        mRawTab->setfilename(DatFileName);

        char tmpStr[2048];

        strcpy(tmpStr,DatFileName.toLatin1().data());

        char *tmp=tmpStr;
        char *prev1=NULL;
        char *prev2=NULL;

        while((tmp = strstr(tmp,"/"))){
            prev2=prev1;
            prev1=tmp;
            tmp++;
        }
        if(prev2 && prev2[1] == 'X'){
            int X=0;
            int Y=0;
            prev2[0]=0;
            prev2++;
            // prev2 points to the block
            sscanf(prev2,"X%d_Y%d",&X,&Y);
            RawExpmtDir = tmpStr;
            RawExpmt_X = X;
            RawExpmt_Y = Y;
            SpatialPlot::SetBlockCoord(RawExpmt_X,RawExpmt_Y);
            printf("%s: X=%d Y=%d basedir=%s\n",__FUNCTION__,X,Y,tmpStr);
            fflush(stdout);
        }else if(prev2 && prev2[1] == 't'){
            // thumbnail
            thumbnail=1;
            int X=0;
            int Y=0;
            prev2[0]=0;
            RawExpmtDir = tmpStr;
            RawExpmt_X = X;
            RawExpmt_Y = Y;
            SpatialPlot::SetBlockCoord(RawExpmt_X,RawExpmt_Y);
            printf("%s: X=%d Y=%d basedir=%s\n",__FUNCTION__,X,Y,tmpStr);
            fflush(stdout);

        }
        AutoCompletePaths();
        currentChanged(0);
    }
}

void Dialog::browseBfMaskDir()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                               tr("Open BfMask File"), BfMaskFileName,"*.bin*");

    printf("%s: %s\n",__FUNCTION__,fileName.toLatin1().data());
    if (!fileName.isEmpty()) {
        BfMaskFileName=fileName;
        mBfMaskTab->setfilename(BfMaskFileName);
        currentChanged(1);
    }
}

void Dialog::browseNoiseDir()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                               tr("Open Noise File"), NoiseFileName,"NoisePic3.dat");

    printf("%s: %s\n",__FUNCTION__,fileName.toLatin1().data());
    if (!fileName.isEmpty()) {
        NoiseFileName=fileName;
        mNoiseTab->setfilename(NoiseFileName);
        currentChanged(2);
    }
}

void Dialog::browseGainDir()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                               tr("Open Gain File"), GainFileName,"Gain.lsr");

    printf("%s: %s\n",__FUNCTION__,fileName.toLatin1().data());
    if (!fileName.isEmpty()) {
        GainFileName=fileName;
        mGainTab->setfilename(GainFileName);
        currentChanged(3);
    }
}


void Dialog::browseWellsDir()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                               tr("Open Wells File"), WellsFileName,"*.wells");

    printf("%s: %s\n",__FUNCTION__,fileName.toLatin1().data());
    if (!fileName.isEmpty()) {
        WellsFileName=fileName;
        mWellsTab->setfilename(WellsFileName);
        currentChanged(4);

		char tmpStr[2048];
		char *ptr=NULL;
		int found=0;

		strcpy(tmpStr,WellsFileName.toLatin1().data());

		if((ptr = strstr(tmpStr,"/onboard_results/sigproc_results"))){
			*ptr=0;
			found=1;
			// were on the instrument??
		}
		else if((ptr = strstr(tmpStr,"/sigproc_results"))){
			*ptr=0;
			found=1;
			thumbnail=1;
			OutputDir = tmpStr;
			RawExpmtDir = tmpStr + QString("/rawdata/");
			RawExpmt_X = 0;
			RawExpmt_Y = 0;
            SpatialPlot::SetBlockCoord(RawExpmt_X,RawExpmt_Y);
			printf("%s: RawExpmtDir = %s\n",__FUNCTION__,RawExpmtDir.toLatin1().data());
		}

		if(found)
			AutoCompletePaths();

//		char *tmp=tmpStr;
//		char *prev1=NULL;
//		char *prev2=NULL;
//
//		while((tmp = strstr(tmp,"/"))){
//		  prev2=prev1;
//		  prev1=tmp;
//		  tmp++;
//		}
//		if(prev2 && prev2[1] == 'X'){
//		  int X=0;
//		  int Y=0;
//		  prev2[0]=0;
//		  prev2++;
//		  // prev2 points to the block
//		  sscanf(prev2,"X%d_Y%d",&X,&Y);
//		  RawExpmtDir = tmpStr;
//		  RawExpmt_X = X;
//		  RawExpmt_Y = Y;
//		  printf("%s: X=%d Y=%d basedir=%s\n",__FUNCTION__,X,Y,tmpStr);
//		  AutoCompletePaths();
//		  fflush(stdout);
//		}
    }
}

void Dialog::browseBamDir()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                               tr("Open Bam File"), BamFileName,"*.bam");

    printf("%s: %s\n",__FUNCTION__,fileName.toLatin1().data());
    if (!fileName.isEmpty()) {
        QFileInfo file(fileName);
        //file.absoluteDir();
        BamFileName=file.absolutePath();
        mAlignmentTab->setfilename(BamFileName);
        currentChanged(5);
    }
}

void Dialog::currentChanged(int index)
{
    QString MainString = "Ion Dat Explorer ";
    if(index == 0)
        MainString += "(" + DatFileName + ")";
    else if(index == 1)
        MainString += "(" + BfMaskFileName + ")";
    else if(index == 2)
        MainString += "(" + WellsFileName + ")";
    else if(index == 3)
        MainString += "(" + BamFileName + ")";

    setWindowTitle(MainString);

}

void Dialog::AutoCompletePaths()
{
    qDebug() << __PRETTY_FUNCTION__ << ": RawExpmtDir=" << RawExpmtDir;
    qDebug() << __PRETTY_FUNCTION__ << ": OutputDir=" << OutputDir;

    if(RawExpmtDir != "" && OutputDir == ""){
        // try to guess the output dir..
        QString opdir=RawExpmtDir + "/sigproc_results/";
        QFileInfo chk1(opdir);
        if(chk1.exists() && chk1.isDir()){
            OutputDir=opdir;
            qDebug() << __PRETTY_FUNCTION__ << ": OutputDir=" << OutputDir;
        }
    }
    if(RawExpmtDir != "" && OutputDir == "" && !thumbnail){
        // try to guess the output dir..
        QString opdir=RawExpmtDir + "/onboard_results/sigproc_results/";
        QFileInfo chk1(opdir);
        if(chk1.exists() && chk1.isDir()){
            OutputDir=opdir;
            qDebug() << __PRETTY_FUNCTION__ << ": OutputDir=" << OutputDir;
        }
    }
    if(RawExpmtDir != "" && OutputDir == ""){
        // try to guess the output dir..
        QString tmp = RawExpmtDir;
        int pos = tmp.indexOf("/rawdata");
        if(pos){
            tmp.replace("/rawdata","");
            if(tmp != RawExpmtDir){
                QString opdir=tmp + "/sigproc_results/";
                QFileInfo chk1(opdir);
                if(chk1.exists() && chk1.isDir()){
                    OutputDir=tmp;
                    qDebug() << __PRETTY_FUNCTION__ << ": OutputDir=" << OutputDir;
                }
            }
        }
    }

    QString BlockName = "X" + QString::number(RawExpmt_X) +
            "_Y" + QString::number(RawExpmt_Y);
	if(thumbnail)
		BlockName="thumbnail";

    qDebug() << __PRETTY_FUNCTION__ << ": BlockName=" << BlockName;

	if(DatFileName == "" &&
       RawExpmtDir != "" &&
       RawExpmt_X != -1 &&
       RawExpmt_Y != -1){
        // auto-fill in the file
       QString tmp = RawExpmtDir + "/" + BlockName + "/acq_0000.dat";
       QFileInfo check_file(tmp);
       if (check_file.exists() && check_file.isFile()) {
           DatFileName = tmp;
       }
    }

    if(NoiseFileName == "" &&
       RawExpmtDir != ""){
        // auto-fill in the file
        QString tmp = RawExpmtDir + "/NoisePic3.dat";
        QFileInfo check_file(tmp);
        if (check_file.exists() && check_file.isFile()) {
            NoiseFileName = tmp;
        }
    }

    if(GainFileName == "" &&
       RawExpmtDir != "" &&
       RawExpmt_X != -1 &&
       RawExpmt_Y != -1){
        // auto-fill in the file
        QString tmp = RawExpmtDir + "/" + BlockName + "/Gain.lsr";
        QFileInfo check_file(tmp);
        if (check_file.exists() && check_file.isFile()) {
            GainFileName = tmp;
        }
    }

    if(BfMaskFileName == "" &&
       RawExpmtDir != "" &&
       RawExpmt_X != -1 &&
       RawExpmt_Y != -1){
        // auto-fill in the file
       QString tmp  = OutputDir + "/block_" + BlockName + "/bfmask.bin";
        QFileInfo check_file(tmp);
        if (check_file.exists() && check_file.isFile()) {
            BfMaskFileName = tmp;
        }
        else{
            tmp  = OutputDir + "/sigproc_results/bfmask.bin";
             QFileInfo check_file2(tmp);
             if (check_file2.exists() && check_file2.isFile()) {
                 BfMaskFileName = tmp;
             }
        }
    }

    if(WellsFileName == "" &&
       RawExpmtDir != "" &&
       RawExpmt_X != -1 &&
       RawExpmt_Y != -1){
        // auto-fill in the file
       QString tmp  = OutputDir + "/block_" + BlockName + "/1.wells";
       if(thumbnail){
           tmp = OutputDir + "/sigproc_results/1.wells";
       }
        QFileInfo check_file(tmp);
        if (check_file.exists() && check_file.isFile()) {
            WellsFileName = tmp;
        }
    }

    if(BamFileName == "" && OutputDir != ""){
        QString tmpName=OutputDir;
        if(RawExpmtDir != ""){
            // see if rawdata is in the file path..
            const char *tmp = RawExpmtDir.toUtf8().constData();
            char tmpCname[4096];
            strcpy(tmpCname,tmp);
            char *ptr = strstr(tmpCname,"/rawdata");
            if(ptr){
                *ptr=0;
                tmpName=tmpCname;
            }
        }

        BamFileName = tmpName;
    }



    QString tmp="";

    tmp = "&Open Dat file";
    if(DatFileName != ""){
        tmp += " (" + DatFileName + ")";
    }
    OpenDatAct->setText(tmp);
    mRawTab->setfilename(DatFileName);

    tmp = "&Open BfMask file";
    if(BfMaskFileName != ""){
        tmp += " (" + BfMaskFileName + ")";
    }
    OpenBfMaskAct->setText(tmp);
    mBfMaskTab->setfilename(BfMaskFileName);

    tmp = "&Open Noise file";
    if(NoiseFileName != ""){
        tmp += " (" + NoiseFileName + ")";
    }
    OpenNoiseAct->setText(tmp);
    mNoiseTab->setfilename(NoiseFileName);

    tmp = "&Open Gain file";
    if(GainFileName != ""){
        tmp += " (" + GainFileName + ")";
    }
    OpenGainAct->setText(tmp);
    mGainTab->setfilename(GainFileName);

    tmp = "&Open Wells file";
    if(WellsFileName != ""){
        tmp += " (" + WellsFileName + ")";
    }
    OpenWellsAct->setText(tmp);
    mWellsTab->setfilename(WellsFileName);

    tmp = "&Open Bam file";
    if(BamFileName != ""){
        tmp += " (" + BamFileName + ")";
    }
    OpenBamAct->setText(tmp);
    mAlignmentTab->setfilename(BamFileName);
}


