/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "AlignmentSpatial.h"
#include <iostream>
#include <string>
#include <algorithm>
#include "Calibration/FlowAlignment.h"
#include "ionstats/ionstats.h"

using namespace std;
using namespace BamTools;


extern int parseAlignment(
  BamAlignment &          alignment,
  ReadAlignmentErrors &   base_space_errors,
  ReadAlignmentErrors &   flow_space_errors,
  map<string, string> &   flow_orders,
  string &                read_group,
  const map<char,char> &  reverse_complement_map,
  bool                    evaluate_flow,
  unsigned int            max_flows,
  bool                    evaluate_hp,
  bool &                  invalid_read_bases,
  bool &                  invalid_ref_bases,
  bool &                  invalid_cigar,
  vector<char> &          ref_hp_nuc,
  vector<uint16_t> &      ref_hp_len,
  vector<int16_t> &       ref_hp_err,
  vector<uint16_t> &      ref_hp_flow,
  vector<uint16_t> &      zeromer_insertion_flow,
  vector<uint16_t> &      zeromer_insertion_len
);
extern void initialize_reverse_complement_map(map<char, char> &rc);
extern void getReadGroupInfo(const BamReader &input_bam, map< string, int > &read_groups, map< string, string > &flow_orders, unsigned int &max_flow_order_len, map< string, string > &key_bases, map< string, int > &key_len, const string &seq_key, const string &skip_rg_suffix);

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

void AlignmentSpatial::setSlider(int per)
{
    QualityLimit = per;
    render();
}


void AlignmentSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    if(txt == "Mask UnAligned")
        SetMask(0,state);
    else if(txt == "Mask Aligned")
        SetMask(1,state);
    else if(txt == "Show NoBarcode")
        ShowNoBarcodes=state;
    else if(txt == "Show Sequence")
        ShowSequence=state;
    else if(txt == "Show Query/Align")
        showQueryBases=state;
    else if(txt == "InvertQuality")
        InvertQuality=state;
    render();
}

// mask the raw tab output to show only aligned or unaligned data
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
                if((aligned && (sequenceLenOut[i] > 0)) || (!aligned && (sequenceLenOut[i] == 0)))
                    mMask[i] |= 0x1;
            }
        }
    }
}

void AlignmentSpatial::DoubleClick_traces(int x, int y, int ts, float val)
{
    qDebug() << __PRETTY_FUNCTION__ << ": Y" << QString::number(y) << "_X" << QString::number(x) << " TS " << ts << "val " << QString::number(val);
    curDisplayFlow = ts; // switch raw display to this flow
    GblTracesVer++; // force it to call render

}

void AlignmentSpatial::doConvert(int &loading)
{

    if(fname == ""){
        if(sequenceLenOut){
            free(sequenceLenOut);
            sequenceLenOut=NULL;
        }
        last_fname = _fname;
        traces_len=0;
    }
    if (last_fname != _fname && !loading){
        loading=0;
        // load a new BAM file
        printf("triggered %s %s\n",last_fname.toLatin1().data(),_fname.toLatin1().data());
        fflush(stdout);
        last_fname = _fname;
        if(sequenceLenOut)
            free(sequenceLenOut);
        sequenceLenOut=NULL;
        if(SequenceQueryBasesOut)
            free(SequenceQueryBasesOut);
        SequenceQueryBasesOut=NULL;
        if(SequenceAlignedBasesOut)
            free(SequenceAlignedBasesOut);
        SequenceAlignedBasesOut=NULL;

        int nbytes;
        // region size
        nbytes=sizeof(sequenceLenOut[0])*rows*cols;
        sequenceLenOut = (int32_t *)malloc(nbytes); memset(sequenceLenOut,0,nbytes);
        nbytes=sizeof(SequenceQueryBasesOut[0])*rows*cols;
        SequenceQueryBasesOut   = (char **)malloc(nbytes); memset(SequenceQueryBasesOut,0,nbytes);
        nbytes=sizeof(SequenceQueryBasesOut[0])*rows*cols;
        SequenceAlignedBasesOut = (char **)malloc(nbytes); memset(SequenceAlignedBasesOut,0,nbytes);
        nbytes=sizeof(SequenceMapQuality[0])*rows*cols;
        SequenceMapQuality = (uint16_t *)malloc(nbytes); memset(SequenceMapQuality,0,nbytes);
        nbytes=sizeof(SequenceMapErrors[0])*rows*cols;
        SequenceMapErrors = (uint32_t **)malloc(nbytes); memset(SequenceMapErrors,0,nbytes);

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

//        sprintf(cmd,"grep Flows %s/explog.txt",fname);
//        popen("grep Flows %")

        _fname = fname;//QString(tmpName);

        if(_fname.contains(".bam")){
            ReadFile((char *)_fname.toUtf8().constData(),0);
        }else{
            QDir dir(_fname);
            char EntryName[4096];
            qDebug() << "Dir: " << _fname;
            if(flows == 0){
                char cmd[2048];
                sprintf(cmd,"bash -c 'grep \"Flows:\" %s/explog.txt | head -n 1 | cut -d\":\" -f 2'",_fname.toUtf8().constData());
                qDebug() << "cmd = " << cmd;
                FILE *fp = popen(cmd,"r");
                if(fp){
                    fscanf(fp,"%d",&flows);
                    qDebug() << __FUNCTION__ << "got " << flows << " flows";
                    pclose(fp);
                }
            }
            foreach(QFileInfo item, dir.entryInfoList() ){
                if(item.isFile()){
                    QString ifn = item.canonicalFilePath();
                    strcpy(EntryName,ifn.toUtf8().constData());
    //                qDebug() << "File: " << ifn << " - " << EntryName;
    //                printf("  File: %s\n",EntryName);
                    fflush(stdout);
                    char *bamPtr = strstr(EntryName,".bam");
                    if(bamPtr && bamPtr[4] == 0){
                        printf("Loading: %s\n",EntryName);
                        fflush(stdout);
                        ReadFile(EntryName,0);
                    }
                }
            }
            // add the nobarcode reads as well..
            _fname += "/basecaller_results/nomatch_rawlib.basecaller.bam";
            strcpy(EntryName,_fname.toUtf8().constData());
            printf("Loading: %s\n",EntryName);
            fflush(stdout);
            ReadFile(EntryName,0);
        }

//        ReadFile(fn);

        traces_len=flows;
        for(int i=0;i<traces_len;i++)
            traces_ts[i]=i;

//        loading=0;
        printf("\ndone\n\n");
    }
}

void AlignmentSpatial::ReadFile(char *fn, int adder)
{
    static uint gbl=0;
    if(threads[gbl] == NULL)
        threads[gbl]=new AlignmentRdrThread();
    threads[gbl++]->LoadFile(cols, rows, Block_X, Block_Y, sequenceLenOut, SequenceQueryBasesOut, SequenceAlignedBasesOut,
                             SequenceMapQuality, SequenceMapErrors, flowOrder, &flowOrderLen, fn, &loading, adder);
    if(gbl > (sizeof(threads)/sizeof(threads[0])))
        gbl=0;

}

float AlignmentSpatial::Get_Data(int frame, int y, int x)
{
    (void)frame;
    float rc = 0;

    char *seq = ApplyFilters(x,y);
    if(seq){
        rc = strlen(seq);//sequenceLenOut[idx];
    }
    return rc;
}


void AlignmentSpatial::CustomTracePlotAdder(double &xmin, double &xmax, double &ymin, double &ymax)
{
    static QCPItemText *seqText[MAX_NUM_TRACES]={NULL};
    static QCPItemText *seqText2[MAX_NUM_TRACES]={NULL};
    if(ymin > -0.5)
        ymin=-0.5;
    if(ymax < 5)
        ymax=5;

    if (traces_len)
    {
        for(int trc=0;trc<MAX_NUM_TRACES;trc++)
        {
            int idx=traces[trc].y*cols+traces[trc].x;
            double x_coord = xmin+0.05*(xmax-xmin);//0.2;
            double y_coord = ymin+0.8*(double)(MAX_NUM_TRACES - trc)*(ymax-ymin)/(double)MAX_NUM_TRACES;//MAX_NUM_TRACES - trc;

//                printf("%s: %d y_coord=%lf ymax=%lf ymin=%lf %d %d\n",__FUNCTION__,trc,y_coord,ymax,ymin,(SequenceQueryBasesOut && SequenceQueryBasesOut[idx]?1:0),(SequenceAlignedBasesOut && SequenceAlignedBasesOut[idx]?1:0));
            char *seq = ApplyFilters(traces[trc].x,traces[trc].y);

            if(seq){
                if(seqText[trc] == NULL){
                    //QPen qp;

                    uint xcolorIdx= trc*ColormapSize/MAX_NUM_TRACES;
                    uint xcolor=colormap[xcolorIdx % ColormapSize];

                    //qp.setColor(QColor(qRed(xcolor),qGreen(xcolor),qBlue(xcolor)));
                    QFont myfont("QFont::Courier", 7);
                    myfont.setStyleHint(QFont::TypeWriter);

                    seqText[trc] = new QCPItemText(_mTracePlot);
                    _mTracePlot->addItem(seqText[trc]);
                    seqText[trc]->setColor(QColor(qRed(xcolor),qGreen(xcolor),qBlue(xcolor)));
                    seqText[trc]->setFont(myfont);
                    seqText[trc]->setPositionAlignment(Qt::AlignLeft|Qt::AlignVCenter);
                }
                if(seqText2[trc] == NULL){
                    //QPen qp;

                    uint xcolorIdx= trc*ColormapSize/MAX_NUM_TRACES;
                    uint xcolor=colormap[xcolorIdx % ColormapSize];

                    //qp.setColor(QColor(qRed(xcolor),qGreen(xcolor),qBlue(xcolor)));
                    QFont myfont("QFont::Courier", 7);
                    myfont.setStyleHint(QFont::TypeWriter);

                    seqText2[trc] = new QCPItemText(_mTracePlot);
                    _mTracePlot->addItem(seqText2[trc]);
                    seqText2[trc]->setColor(QColor(qRed(xcolor),qGreen(xcolor),qBlue(xcolor)));
                    seqText2[trc]->setFont(myfont);
                    seqText2[trc]->setPositionAlignment(Qt::AlignLeft|Qt::AlignVCenter);
                }
                seqText[trc]->position->setCoords(x_coord, y_coord);
                seqText2[trc]->position->setCoords(x_coord, y_coord+30);


                //printf("%s: setting text to %s",__FUNCTION__,SequenceAlignedBasesOut[idx]);
                char localStr[250];
                sprintf(localStr,"QB:%d:%.200s",strlen(SequenceQueryBasesOut[idx]),SequenceQueryBasesOut[idx]);
                seqText[trc]->setText(localStr);
                sprintf(localStr,"AB:%d:%.200s",strlen(SequenceAlignedBasesOut[idx]),SequenceAlignedBasesOut[idx]);
                seqText2[trc]->setText(localStr);

            }else{
                if(seqText[trc])
                    seqText[trc]->setText("");
                if(seqText2[trc])
                    seqText2[trc]->setText("");
            }
        }
    }
}

char *AlignmentSpatial::ApplyFilters(int x, int y)
{
    char *rc=NULL;
    int idx = y*cols + x;

    if((x >=0) && (y >= 0) && x < cols && y<rows && SequenceQueryBasesOut && SequenceAlignedBasesOut){

        int MapQualityFilter = ShowNoBarcodes || (((!InvertQuality && SequenceMapQuality[idx] >= QualityLimit) ||
                                (InvertQuality && SequenceMapQuality[idx] < QualityLimit)));
        int ShowNoBarcodeFilter = (ShowNoBarcodes && !SequenceAlignedBasesOut[idx]) ||
                (!ShowNoBarcodes && SequenceAlignedBasesOut[idx]);
        char *seq = SequenceAlignedBasesOut[idx];
        if(showQueryBases)
            seq = SequenceQueryBasesOut[idx];

        if(seq && MapQualityFilter && ShowNoBarcodeFilter){
            rc = seq;
        }
    }
    return rc;
}

void AlignmentSpatial::UpdateTraceData()
{
    LocalTracesVer = GblTracesVer;

    traces_len=flows;
    for(int i=0;i<traces_len;i++)
        traces_ts[i]=i;

    memset(traces_Val,0,sizeof(traces_Val));
    memset(traces_highlited,0,sizeof(traces_highlited));

    // copy from out to traces
    for(int trc=0;trc<MAX_NUM_TRACES;trc++)
    {
        char *seq = ApplyFilters(traces[trc].x,traces[trc].y);
        int idx=traces[trc].y*cols+traces[trc].x;
        if(seq && flowOrder[0]){
            int seqLen=strlen(seq);
            int lflow=0;
            int cseq=0;
            while(lflow<traces_len && cseq < seqLen){
                while(seq[cseq] == flowOrder[lflow % flowOrderLen]){
                    traces_Val[trc][lflow] += 1;
                    cseq++;
                }
                lflow++;
            }
            int errIdx=0;
            while(errIdx < 10 && SequenceMapErrors[idx] && SequenceMapErrors[idx][errIdx] ){
                int tflow = SequenceMapErrors[idx][errIdx] & 0xff;
                if(tflow > 0 && tflow < traces_len)
                    traces_highlited[trc][tflow]=1;
                errIdx++;
            }
        }
    }
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
void AlignmentRdrThread::LoadFile(int _w, int _h, int _startX, int _startY, int32_t *_out, char **_outSeq,
                                  char **_outAlignSeq, uint16_t *_outMapQuality, uint32_t **_outErrSeq,
                                  char *_outFlowOrder, int *_outflowOrderLen, char *_fileName, int *_loading,
                                  int _adder)
{
    w=_w;
    h=_h;
    startX=_startX;
    startY=_startY;
    out=_out;
    outSeq = _outSeq;
    outAlignSeq = _outAlignSeq;
    outErrSeq = _outErrSeq;
    outMapQuality = _outMapQuality;
    outFlowOrder = _outFlowOrder;
    outFlowOrderLen = _outflowOrderLen;
    loading=_loading;

    adder=_adder;

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
        barcodeHeader[0]=0;
        if(strstr(fileName,"nomatch") == NULL){
            char cmd[2048];
            sprintf(cmd,"bash -c 'file=$(basename %s); dir=$(dirname %s); grep ${file/_rawlib.bam/} ${dir}/barcodeList.txt | cut -d \",\" -f 3'",fileName,fileName);
            qDebug() << "cmd = < " << cmd;
            FILE *fp = popen(cmd,"r");
            if(fp){
                fscanf(fp,"%s",barcodeHeader);
                qDebug() << "got barcode header for " << fileName << " of " << barcodeHeader;
                pclose(fp);
            }
        }

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
            printf("%s: sequenceLenOut=%p\n",__FUNCTION__,out);
            printf("%s: SequenceAlignedBasesOut=%p\n",__FUNCTION__,outSeq);


            map< string, int > read_groups;
            map< string, string > flow_orders;
            map< string, string > key_bases;
            map< string, int > key_len;
            unsigned int max_flow_order_len=0;
            string seq_key="tacg";
            string skip_rg_suffix="";
            getReadGroupInfo(reader,read_groups,flow_orders,max_flow_order_len,key_bases,key_len,seq_key,skip_rg_suffix);

            for(map< string, string >::iterator flow_order_it=flow_orders.begin(); flow_order_it != flow_orders.end(); ++flow_order_it) {
              if(outFlowOrder[0] == 0 && flow_order_it->second.length() > 0) {
                strcpy(outFlowOrder,flow_order_it->second.c_str());
                *outFlowOrderLen = strlen(outFlowOrder);
              }
            }


            BamAlignment al;
            while ( !abort && reader.GetNextAlignment(al) ) {
                int x,y;
                ion_readname_to_xy(al.Name.c_str(), &x, &y);
                int region_x = x - startX;
                int region_y = y - startY;
                if (region_x >= 0 && region_x < w && region_y >=0 && region_y < h){
                    int idx=region_y*w + region_x;
                    const char *QuerySeqStr=al.QueryBases.c_str();
                    const char *AlignSeqStr=al.AlignedBases.c_str();



                    out[idx] = al.AlignedBases.length() + adder;
                    outMapQuality[idx] = al.MapQuality;
                    if(outSeq[idx]){
                        free(outSeq[idx]);
                        outSeq[idx]=NULL;
                    }
                    if(outAlignSeq[idx]){
                        free(outAlignSeq[idx]);
                        outAlignSeq[idx]=NULL;
                    }
                    if(outErrSeq[idx]){
                        free(outErrSeq[idx]);
                        outErrSeq[idx]=NULL;
                    }
                    if(strlen(QuerySeqStr)){
                        string tmpStr="TCAG";
                        //if(strlen(AlignSeqStr)==0){
                            tmpStr += barcodeHeader;
                        //}
                        tmpStr += QuerySeqStr;
                        outSeq[idx] = strdup(tmpStr.c_str());
                    }
                    if(strlen(AlignSeqStr)){
                        string tmpStr="TCAG";
                        tmpStr += barcodeHeader;
                        tmpStr += AlignSeqStr;
                        outAlignSeq[idx] = strdup(tmpStr.c_str());
                        outErrSeq[idx] = (uint32_t *)malloc(sizeof(uint32_t)*MAX_ERROR_FLOWS);
                        memset(outErrSeq[idx],0,sizeof(uint32_t)*MAX_ERROR_FLOWS);
//                        printf("%s: idx=%d alignSeqStr=%s\n",__FUNCTION__,idx,outAlignSeq[idx]);


                        ReadAlignmentErrors base_space_errors;
                        ReadAlignmentErrors flow_space_errors;
                        //map<string, string> flow_orders;
                        string              read_group;
                        map<char,char>   reverse_complement_map;
                        bool                   invalid_read_bases;
                        bool                   invalid_ref_bases;
                        bool                   invalid_cigar;
                        vector<char>           ref_hp_nuc;
                        vector<uint16_t>       ref_hp_len;
                        vector<int16_t>        ref_hp_err;
                        vector<uint16_t>       ref_hp_flow;
                        vector<uint16_t>       zeromer_insertion_flow;
                        vector<uint16_t>       zeromer_insertion_len;

                        initialize_reverse_complement_map(reverse_complement_map);
                        int parserc = parseAlignment(
                          al,               // The BAM record to be parsed
                          base_space_errors,       // Returns the errors in base space
                          flow_space_errors,       // If evaluate_flow is TRUE, returns the errors in flow space
                          flow_orders,             // Specifies the flow orders for each read group
                          read_group,              // Used to return the read group for the BAM record
                          reverse_complement_map,  // Defines the rev comp for every base
                          1,           // Specifies if flows should be analyzed & returned
                          400,               // Max flow to analyze
                          0,             // Specifies if per-HP results should be analyzed & returned
                          invalid_read_bases,      // Will return TRUE if invalid (non-IUPAC) bases encountered in the read
                          invalid_ref_bases,       // Will return TRUE if invalid (non-IUPAC) bases encountered in the ref
                          invalid_cigar,           // Will return TRUE if lengths implied by CIGAR and by alignment do not match
                          ref_hp_nuc,              // If evaulate_hp is TRUE, specifies ref hp nucleotides
                          ref_hp_len,              // If evaulate_hp is TRUE, specifies ref hp lengths
                          ref_hp_err,              // If evaulate_hp is TRUE, specifies the read HP length errors (positive means overcall)
                          ref_hp_flow,             // If evaulate_hp and evaluate_flow are TRUE, specifies flow to which each ref hp aligns
                          zeromer_insertion_flow,  // If evaluate_flow and evaluate_hp are TRUE, specifies all flows with an insertion in a reference zeromer
                          zeromer_insertion_len    // If evaluate_flow and evaluate_hp are TRUE, specifies the lengths of the zeromer insertions
                        );
                        if(parserc != 0){
                            printf("Failed to find rg < %s > flow order\n",read_group.c_str());
                        }else{
                            const vector<uint16_t> &insertions = flow_space_errors.ins();
                            const vector<uint16_t> &deletions  = flow_space_errors.del();
                            // mark each bad flow as an ins or del
                            int errIdx=0;
                            for(uint err=0;err<insertions.size() && errIdx < MAX_ERROR_FLOWS ;err++){
                                outErrSeq[idx][errIdx++] = insertions[err] | 0x1000;
                            }
                            for(uint err=0;err<deletions.size() && errIdx < MAX_ERROR_FLOWS;err++){
                                outErrSeq[idx][errIdx++] = deletions[err] | 0x2000;
                            }
                        }
                    }
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

void AlignmentSpatial::SetTracesToolTip(int y, int x, float ts, float val)
{
    uint16_t mapQual=0;
    if(SequenceMapQuality)
        mapQual = SequenceMapQuality[y*cols+x];
    _mTracePlot->setToolTip(QString("Y%1_X%2: (%3,%4)  %5 Qual=%6").arg(y).arg(x).arg(ts).arg(val).arg(flowOrder[((int)ts) % flowOrderLen]).arg(mapQual));

}




