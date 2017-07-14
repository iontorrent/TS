/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "AlignmentSpatial.h"
#include <iostream>
#include <string>
#include <algorithm>
#include "Calibration/FlowAlignment.h"

using namespace std;
using namespace BamTools;


AlignmentSpatial::AlignmentSpatial(QWidget *parent): SpatialPlot(parent)
{
}


// copied from ion_util.c
int ion_readname_to_xy(const char *readname, int *x, int *y)
{
  int i, val, state;
  /* states:
     0 - skipping over read name (before first colon)
     1 - reading in x value (before second colon)
     2 - reading in y value (after second colon)
     */
  for(i=val=state=0;'\0' != readname[i];i++) {
      if(':' == readname[i]) {
          if(1 == state) {
              (*y) = val;
          }
          state++;
          val = 0;
      }
      else if('0' <= readname[i] && readname[i] <= '9') {
          val *= 10;
          val += (int32_t)(readname[i] - '0');
      }
  }
  if(2 == state) {
      (*x) = val;
      return 1;
  }
  else {
      return 0;
  }
}

void AlignmentSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    if(txt == "Mask UnAligned")
        SetMask(0,state);
    else if(txt == "Mask Aligned")
        SetMask(1,state);

    render();
}

void AlignmentSpatial::SetMask(int aligned, int state)
{
    if(mMask){

        // set the global mask bits for aligned or unaligned reads
        if(!state){
            // clear the mask bits for all reads
            for(int i=0;i<(rows*cols);i++)
                mMask[i] &= ~0x1;
        }else{
            for(int i=0;i<(rows*cols);i++){
                if((aligned && (out[i] > 0)) || (!aligned && (out[i] == 0)))
                    mMask[i] |= 0x1;
            }
        }
    }
}

void AlignmentSpatial::doConvert(int &loading)
{

    if(fname == ""){
        free(out);
        out=NULL;
        last_fname = _fname;
        traces_len=0;
    }
    if (last_fname != _fname && !loading){
        loading=0;
        // load a new BAM file
        printf("triggered %s %s\n",last_fname.toLatin1().data(),_fname.toLatin1().data());
        fflush(stdout);
        last_fname = _fname;
        if(out)
            free(out);
        out=NULL;

        // region size
        int nbytes=sizeof(out[0])*rows*cols;
        out = (int32_t *)(int32_t *)malloc(nbytes);

//        char tmpName[4096];
//        strcpy(tmpName,fname.toStdString().data());
//        char *ptr = tmpName;
//        char *last=NULL;
//        while((ptr = strstr(ptr,"/"))){
//        	last = ptr;
//        	ptr++;
//        }
//        if(last)
//            *last = 0;

        _fname = fname;//QString(tmpName);
        QDir dir(_fname);
        qDebug() << "Dir: " << _fname;
        foreach(QFileInfo item, dir.entryInfoList() ){
            if(item.isFile()){
                QString ifn = item.canonicalFilePath();
                char EntryName[4096];
                strcpy(EntryName,ifn.toUtf8().constData());
                qDebug() << "File: " << ifn << " - " << EntryName;
//                printf("  File: %s\n",EntryName);
                fflush(stdout);
                char *bamPtr = strstr(EntryName,".bam");
                if(bamPtr && bamPtr[4] == 0){
                    printf("Loading: %s\n",EntryName);
                    fflush(stdout);
                    ReadFile(EntryName);
                }
            }
        }

//        ReadFile(fn);

        traces_len=1;
        for(int i=0;i<traces_len;i++)
            traces_ts[i]=i;

//        loading=0;
        printf("\ndone\n\n");
    }
}

void AlignmentSpatial::ReadFile(char *fn)
{
    static uint gbl=0;
    if(threads[gbl] == NULL)
        threads[gbl]=new AlignmentRdrThread();
    threads[gbl++]->LoadFile(cols, rows, Block_X, Block_Y, out, fn, &loading, &seqIndx, &alignments);
    if(gbl > (sizeof(threads)/sizeof(threads[0])))
        gbl=0;

#if 0
    BamReader reader;
    if ( !reader.Open(fn) ) {
        printf("Could not open input BAM file %s\n",fn);
        loading=0;
        return;
    }

    // region position
    int RawExpmt_Y=Block_Y;
    int RawExpmt_X=Block_X;

    BamAlignment al;
    while ( reader.GetNextAlignment(al) ) {
        int x,y;
        ion_readname_to_xy(al.Name.c_str(), &x, &y);
        int region_x = x - RawExpmt_X;
        int region_y = y - RawExpmt_Y;
        if (region_x >= 0 && region_x < cols && region_y >=0 && region_y < rows){
            out[region_y*cols + region_x] = al.AlignedBases.length();
            //cout << x << ":" << y << " length " << al.Length << endl;
        }
    }
    reader.Close();
#endif
}

float AlignmentSpatial::Get_Data(int frame, int y, int x)
{
    (void)frame;
    float rc = 0;

    if(out && y < rows && x < cols){
        rc = out[/*frame*rows*cols +*/ y*cols + x];
    }
    return rc;
}

BamAlignment* AlignmentSpatial::Get_Alignment(int y, int x)
{
    int found = std::find(seqIndx.begin(), seqIndx.end(), y*cols + x) - seqIndx.begin();
    if (found < (int)seqIndx.size()){
        return &alignments[found];
    }
    return NULL;
}

void AlignmentSpatial::Get_ReferenceSeq(BamAlignment *al, string &qseq, string &tseq){
    // tseq: refernce (target) bases for aligned portion of the read
    // qseq: read (query) bases for aligned portion of the read
    string md;
    string tseq_bases;
    string qseq_bases;
    string pretty_tseq;
    string pretty_qseq;
    string pretty_aln;
    unsigned int left_sc, right_sc;

    al->GetTag("MD", md);
    RetrieveBaseAlignment(al->QueryBases, al->CigarData, md, tseq_bases, qseq_bases,
                          pretty_tseq, pretty_qseq, pretty_aln, left_sc, right_sc);
    qseq = pretty_qseq;
    tseq = pretty_tseq;
}

void AlignmentSpatial::UpdateTracePlot()
{
    printf("%s\n",__FUNCTION__);

    QFont myfont("QFont::Courier", 8);
    myfont.setStyleHint(QFont::TypeWriter);
    _mTracePlot->clearItems();

    if (traces_len)
    {
        for(int trc=0;trc<MAX_NUM_TRACES;trc++)
        {
            if((traces[trc].x >=0) && (traces[trc].y > 0))
            {
                // draw graph line
                QVector<double> x(2);
                QVector<double> y(2);
                x[0] = 0; x[1] = 0.2;
                y[0] = y[1] = MAX_NUM_TRACES - trc;

                _mTracePlot->graph(trc)->setData(x, y);
                QString name="Y" + QString::number(traces[trc].x) + "_X" + QString::number(traces[trc].y);
                _mTracePlot->graph(trc)->setName(name);
                _mTracePlotSet[trc]=1;

                // add sequence text
                BamAlignment *al= Get_Alignment(traces[trc].y, traces[trc].x);
                if (al){
                    QCPItemText *seqText = new QCPItemText(_mTracePlot);
                    _mTracePlot->addItem(seqText);
                    seqText->position->setCoords(x[1], y[0]);
                    seqText->setFont(myfont);
                    seqText->setPositionAlignment(Qt::AlignLeft|Qt::AlignVCenter);

                    string qseq, tseq;
                    Get_ReferenceSeq(al, qseq, tseq);
                    string text = qseq + "\n" + tseq;
                    cout << text << endl;
                    seqText->setText(text.c_str());
                }
            }
        }
    }

    _mTracePlot->xAxis->setRange(0, 10);
    _mTracePlot->yAxis->setRange(0, MAX_NUM_TRACES+0.5);
    _mTracePlot->replot();
}


AlignmentRdrThread::AlignmentRdrThread(QObject *parent)
    : QThread(parent)
{
}

//! [0]

//! [1]
AlignmentRdrThread::~AlignmentRdrThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}
//! [1]


//! [2]
void AlignmentRdrThread::LoadFile(int _w, int _h, int _startX, int _startY, int32_t *_out, char *_fileName, int *_loading, vector<int> *_seqIndx, vector<BamAlignment> *_alignments)
{
    w=_w;
    h=_h;
    startX=_startX;
    startY=_startY;
    out=_out;
    loading=_loading;
    seqIndx= _seqIndx;
    alignments= _alignments;

    strcpy(fileName,_fileName);

    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        __sync_add_and_fetch(loading,1);
        start(LowPriority);
    } else {
        //restart = true;
        condition.wakeOne();
    }
}
//! [2]

void AlignmentRdrThread::Abort()
{
    abort=true;
    mutex.unlock();
}

void AlignmentRdrThread::run()
{
//    forever {
//        mutex.lock();
        if(abort)
            return; // application is closed
//        mParent->copyData();
//        mutex.unlock();


        BamReader reader;
        if ( !reader.Open(fileName) ) {
            printf("Could not open input BAM file %s\n",fileName);
            //loading=0;
            //return;
        }
        else{

        // region position
//        int RawExpmt_Y=Block_Y;
//        int RawExpmt_X=Block_X;

            BamAlignment al;
            while ( !abort && reader.GetNextAlignment(al) ) {
                int x,y;
                ion_readname_to_xy(al.Name.c_str(), &x, &y);
                int region_x = x - startX;
                int region_y = y - startY;
                if (region_x >= 0 && region_x < w && region_y >=0 && region_y < h){
                    out[region_y*w + region_x] = al.AlignedBases.length();
//                    seqIndx->push_back(region_y*w + region_x);
//                    alignments->push_back(al);
                }
            }
            reader.Close();
            printf("done loading %s\n",fileName);
        }

        __sync_sub_and_fetch (loading,1);



//        mutex.lock();
//        if (!restart)
//            condition.wait(&mutex);
//        restart = false;
//        mutex.unlock();
//    }
}



