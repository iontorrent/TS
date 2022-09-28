/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */

#include <QPainter>
#include <QKeyEvent>

#include <math.h>

#include "SpatialPlot.h"


//! [0]
const double ZoomInFactor = 0.7;
const double ZoomOutFactor = 1 / ZoomInFactor;
const int ScrollStep = 20;
//! [0]

// window variables
int SpatialPlot::startX=0;
int SpatialPlot::endX=0;
int SpatialPlot::startY=0;
int SpatialPlot::endY=0;
int SpatialPlot::frame=0;
int SpatialPlot::flow=0;

// image variables
int SpatialPlot::rows=0;
int SpatialPlot::cols=0;
int SpatialPlot::frames=0;
int SpatialPlot::flows=0;

int SpatialPlot::Block_X=0;
int SpatialPlot::Block_Y=0;
char *SpatialPlot::mMask = NULL;
float *SpatialPlot::WellsOut=NULL;    // raw calls data
float *SpatialPlot::WellsNormOut=NULL;    // raw calls data
int32_t *SpatialPlot::sequenceLenOut=NULL; // length of alignments
char **SpatialPlot::SequenceQueryBasesOut=NULL;   // called bases
char **SpatialPlot::SequenceAlignedBasesOut=NULL; // aligned bases
uint16_t *SpatialPlot::SequenceMapQuality=NULL;
uint32_t **SpatialPlot::SequenceMapErrors=NULL;
float *SpatialPlot::NumpyOut=NULL;
float *SpatialPlot::microscopeOut=NULL;    // raw calls data
int SpatialPlot::flowOrderLen=0;
char SpatialPlot::flowOrder[4096]={0};
int SpatialPlot::curDisplayFlow=-1;


TraceDescr_t SpatialPlot::traces[MAX_NUM_TRACES] = {{0,0}};
int SpatialPlot::traces_curSelection=0;
int SpatialPlot::GblTracesVer=0;
uint16_t *SpatialPlot::refMask=NULL; //from BfMaskTab
float *SpatialPlot::gainVals=NULL;   //from GainTab


//! [1]
SpatialPlot::SpatialPlot(QWidget *parent)
    :QWidget(parent)
{

    connect(&thread, SIGNAL(renderedImage(QImage)), this, SLOT(updatePixmap(QImage)));

    setWindowTitle(tr("SpatialPlot"));
#ifndef QT_NO_CURSOR
    setCursor(Qt::CrossCursor);
#endif
    resize(550, 400);
    setMouseTracking( true );
    setFocusPolicy(Qt::WheelFocus);

    for (int i = 0; i < ColormapSize; ++i)
        colormap[i] = rgbFromWaveLength((375.0) + (i * 400.0 / ColormapSize));

    colormap[0]=0; // make the background black

     thread.SetParent(this);

     memset(traces,0,sizeof(traces));
     for(int trc=0;trc<MAX_NUM_TRACES;trc++)
     {
         if(trc < 1)
            traces[trc].y = traces[trc].x = trc;
         else
            traces[trc].y = traces[trc].x = -1;
     }


}
//! [1]

SpatialPlot::~SpatialPlot()
{
   thread.Abort();
}

void SpatialPlot::SetOption(QString option, int state)
{
    (void)option;
    (void)state;
}

void SpatialPlot::setfilename(QString fileName)
{
    printf("%s: %s\n",__PRETTY_FUNCTION__,fileName.toLatin1().data());
    fname = fileName;
    render();
}

void SpatialPlot::setSlider(int per)
{
    if(flow_based)
        flow  = flows*per/100;
    else
        frame = frames*per/100;
    render();
}

void SpatialPlot::render()
{
    if(firstShow)
        thread.render();
}

void SpatialPlot::paintEvent(QPaintEvent * /* event */)
{
    if(!firstShow ||
            (lastStartX != startX) ||
            (lastStartY != startY) ||
            (lastEndX   != endX) ||
            (lastEndY   != endY) ||
            //(lastcurDisplayFlow != curDisplayFlow) ||
            (GblTracesVer != LocalTracesVer)){
        thread.render();
    }
    firstShow=1;


    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

//    else if (pixmap.isNull()) {
//        painter.setPen(Qt::white);
//        painter.drawText(rect(), Qt::AlignCenter, tr("Rendering initial image, please wait..."));
//        return;
//    }
    painter.drawPixmap(QPoint(0,0)/*pixmapOffset*/, pixmap);

    if(loading){
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, tr("Loading file, please wait..."));
//        return;
    }

    QFontMetrics metrics = painter.fontMetrics();

    painter.setPen(Qt::white);
    {
        QRect rect(borderWidth,height()-borderHeight,borderWidth-4,borderHeight-4);
        painter.drawText(rect, metrics.leading() + metrics.ascent(), QString::number(startX));
    }
    {
        int nw = metrics.width(QString::number(endX));
        QRect rect(width()-scaleBarWidth-nw,height()-borderHeight,nw,borderHeight-4);
        painter.drawText(rect, metrics.leading() + metrics.ascent(), QString::number(endX));
    }
    {
        int nh = metrics.height();
        QRect rect(1,height()-borderHeight-nh,borderWidth-2,nh);
        painter.drawText(rect, metrics.leading() + metrics.ascent(), QString::number(startY));
    }
    {
        int nh = metrics.height();
        QRect rect(1,2,borderWidth-2,nh);
        painter.drawText(rect, metrics.leading() + metrics.ascent(), QString::number(endY));
    }
    {
        int nh = metrics.height();
        QRect rect(width()-scaleBarWidth+3,height()-nh,scaleBarWidth-3,nh);
        painter.drawText(rect, metrics.leading() + metrics.ascent(), QString::number(minPixVal));
    }
    {
        int nh = metrics.height();
        QRect rect(width()-scaleBarWidth+3,2,scaleBarWidth-3,nh);
        painter.drawText(rect, metrics.leading() + metrics.ascent(), QString::number(maxPixVal));
    }

    if(initialHints)
    {
        QString text = tr("Use mouse wheel or the '+' and '-' keys to zoom. "
                          "Press and hold left mouse button to scroll.");

        int textWidth = metrics.width(text);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 127));
        painter.drawRect((width() - textWidth) / 2 - 5, 0, textWidth + 10, metrics.lineSpacing() + 5);
        painter.setPen(Qt::white);
        painter.drawText((width() - textWidth) / 2, metrics.leading() + metrics.ascent(), text);
    }
}


void SpatialPlot::setRawTrace(int selection)
{
    traces_curSelection = selection;
}


void SpatialPlot::DoubleClick(int x,int y)
{
    printf("%s: %d/%d\n",__FUNCTION__,x,y);
    int trc=traces_curSelection;

    //if(out){
        traces[trc].x = x;
        traces[trc].y = y;
        if(x<0 || y<0){
            // clear this traces values..
            for(int i=0;i<traces_len;i++){
                traces_Val[trc][i]=0;
            }
        }
        GblTracesVer++;
   // }
}

void SpatialPlot::UpdateTraceData()
{
    LocalTracesVer = GblTracesVer;

    memset(traces_highlited,0,sizeof(traces_highlited));

    // copy from out to traces
    int limit=traces_len;
    for(int trc=0;trc<MAX_NUM_TRACES;trc++)
    {
        if((traces[trc].x >=0) && (traces[trc].y > 0))
        {
            for(int idx=0;idx<limit;idx++){
                traces_Val[trc][idx] = Get_Data(idx,
                                                traces[trc].y,traces[trc].x);
            }
        }
    }
}
void SpatialPlot::DisplayHistogram(int *hist, int len, int minVal, int maxVal, int useLog)
{
	{
        QVector<double> x(0);
        QVector<double> y(0);
        for(int bin=0;bin<len;bin++){
        	y.push_back(hist[bin]);
        	x.push_back(minVal + (maxVal-minVal) * bin/len);
        }
        _mTracePlot->graph(0)->setData(x, y);
	}
    for(int trc=1;trc<MAX_NUM_TRACES;trc++){
        if(_mTracePlotSet[trc]){
             QString name="";
             _mTracePlot->graph(trc)->setName(name);
             _mTracePlotSet[trc]=0;
             QVector<double> x(0);
             QVector<double> y(0);
             _mTracePlot->graph(trc)->setData(x, y);
         }
    }
    float ymax=0;
    float ymin=1000;
    float xmax = maxVal;
    float xmin = minVal;

    for(int bin=0;bin<len;bin++){
    	if(hist[bin] > ymax)
    		ymax = hist[bin];
    	if(hist[bin] < ymin)
    		ymin = hist[bin];
    }

    qDebug() << __PRETTY_FUNCTION__ << ": xmin=" << xmin << " xmax=" << xmax << " ymin=" << ymin << " ymax=" << ymax;
    _mTraceRange=ymax-ymin;
    _mTracePlot->xAxis->setRange(xmin, xmax);
    _mTracePlot->yAxis->setRange(ymin, ymax);
  	_mTracePlot->yAxis->setScaleType(useLog?QCPAxis::stLogarithmic:QCPAxis::stLinear);

    _mTracePlot->replot();
}

void SpatialPlot::UpdateTracePlot()
{
    // plot the data in traces
	qDebug() << __PRETTY_FUNCTION__ << ": hist=" << display_histogram;
	if(display_bitsNeeded){
		DisplayHistogram(bitsNeededData,sizeof(bitsNeededData)/sizeof(bitsNeededData[0]),0,16,0);

	}else if(display_histogram){
		DisplayHistogram(histData,sizeof(histData)/sizeof(histData[0]),minPixVal,maxPixVal,1);

	}else if (traces_len)
    {
        double xmin = 0; // set defaults
        double xmax = ((double)traces_ts[traces_len-1]);
        double ymin = 16384;
        double ymax = 0;
        int highlited = 0;
        QVector<double> x_highlited(traces_len/**MAX_NUM_TRACES*/);
        QVector<double> y_highlited(traces_len/**MAX_NUM_TRACES*/);


        for(int trc=0;trc<MAX_NUM_TRACES;trc++)
        {
            if((traces[trc].x >=0) && (traces[trc].y > 0))
            {
                QVector<double> x(traces_len);
                QVector<double> y(traces_len);

                for (int i=0; i<traces_len; i++)
                {
                  x[i] = traces_ts[i];
                  y[i] = traces_Val[trc][i];
                  if(y[i] < ymin)
                      ymin = y[i];
                  if(y[i] > ymax)
                      ymax = y[i];
                  if(traces_highlited[trc][i] && highlited < traces_len){
                      x_highlited[highlited]=traces_ts[i];
                      y_highlited[highlited++]=y[i];
                  }
                }

                // create graph and assign data to it:
                _mTracePlot->graph(trc)->setData(x, y);
                QString name="Y" + QString::number(traces[trc].y) + "_X" + QString::number(traces[trc].x);
                _mTracePlot->graph(trc)->setName(name);
                _mTracePlotSet[trc]=1;
                // set axes ranges, so we see all data:
            }
            else{
                if(_mTracePlotSet[trc]){
                    QString name="";
                    _mTracePlot->graph(trc)->setName(name);
                    _mTracePlotSet[trc]=0;
                    QVector<double> x(0);
                    QVector<double> y(0);
                    _mTracePlot->graph(trc)->setData(x, y);
                }
            }
        }

        _mTracePlot->graph(MAX_NUM_TRACES+1)->setData(x_highlited,y_highlited);

        if(X_Upper_Override != NO_OVERRIDE)
            xmax=X_Upper_Override;
        if(Y_Upper_Override != NO_OVERRIDE)
            ymax=Y_Upper_Override;
        if(X_Lower_Override != NO_OVERRIDE)
            xmin=X_Lower_Override;
        if(Y_Lower_Override != NO_OVERRIDE)
            ymin=Y_Lower_Override;

        CustomTracePlotAdder(xmin,xmax,ymin,ymax);
        {
            // add frame/flow line
            QVector<double> x(2);
            QVector<double> y(2);
            x[0] = x[1] = (double)traces_ts[flow_based?flow:frame];
            y[0] = ymin;
            y[1] = ymax;
            _mTracePlot->graph(MAX_NUM_TRACES)->setData(x, y);
            _mTracePlot->graph(MAX_NUM_TRACES)->setName("");
            _mTracePlot->graph(MAX_NUM_TRACES)->setPen(QPen(Qt::DashLine));

        }
        _mTraceRange=ymax-ymin;
        _mTracePlot->xAxis->setRange(xmin, xmax);
        _mTracePlot->yAxis->setRange(ymin, ymax);
        _mTracePlot->yAxis->setScaleType(QCPAxis::stLinear);
        _mTracePlot->replot();

    }

}

void SpatialPlot::CustomTracePlotAdder(double &xmin, double &xmax, double &ymin, double &ymax)
{
    (void)xmin;
    (void)xmax;
    (void)ymin;
    (void)ymax;
}


void SpatialPlot::setTracePlot(QCustomPlot *mtracePlot, int _flow_based)
{
    _mTracePlot = mtracePlot;
    flow_based=_flow_based;

    double xmin =-1; // set defaults
    double xmax = 1;
    double ymin = 0;
    double ymax = 1;

    // create graph and assign data to it:
    for(int i=0;i<(MAX_NUM_TRACES+2);i++){
        mtracePlot->addGraph();

        if(i < MAX_NUM_TRACES){
            QPen qp;

            uint xcolorIdx= i*ColormapSize/MAX_NUM_TRACES;
            uint xcolor=colormap[xcolorIdx % ColormapSize];

            qp.setColor(QColor(qRed(xcolor),qGreen(xcolor),qBlue(xcolor)));
            mtracePlot->graph(i)->setPen(qp);
        }
        else if(i==(MAX_NUM_TRACES+1)){
            mtracePlot->graph(i)->setScatterStyle(QCPScatterStyle::ssDiamond);
            mtracePlot->graph(i)->setLineStyle(QCPGraph::lsNone);
        }
    }

    mtracePlot->setInteractions(QCP::iRangeZoom | QCP::iSelectPlottables | QCP::iSelectLegend | QCP::iRangeDrag); // allow selection of graphs via mouse click
    // give the axes some labels:
    mtracePlot->xAxis->setLabel("x");
    mtracePlot->yAxis->setLabel("y");
    // set axes ranges, so we see all data:
    mtracePlot->xAxis->setRange(xmin, xmax);
    mtracePlot->yAxis->setRange(ymin, ymax);
    mtracePlot->legend->setVisible(true);
    //mtracePlot->legend->setAlignment(Qt::AlignTop);
    mtracePlot->replot();

    connect(mtracePlot, SIGNAL(mouseMove(QMouseEvent*)), this,SLOT(showPointToolTip_traces(QMouseEvent*)));
    connect(mtracePlot, SIGNAL(mouseDoubleClick(QMouseEvent*)), this,SLOT(mouseDoubleClickEvent_traces(QMouseEvent*)));
    connect(mtracePlot, SIGNAL(axisDoubleClick(QCPAxis *axis, QCPAxis::SelectablePart part, QMouseEvent *event)), this,SLOT(
        axisDoubleClick_traces(QCPAxis *axis, QCPAxis::SelectablePart part, QMouseEvent *event)));
}

void SpatialPlot::showPointToolTip(QMouseEvent *event)
{

    int x = _mTracePlot->xAxis->pixelToCoord(event->pos().x());
    int y = _mTracePlot->yAxis->pixelToCoord(event->pos().y());

    _mTracePlot->setToolTip(QString("%1 , %2").arg(x).arg(y));


}


void SpatialPlot::Save(QTextStream &strm)
{
    // create graph and assign data to it:

    for(int i=0;i<(MAX_NUM_TRACES);i++){

        if((traces[i].x >=0) && (traces[i].y > 0))
        {
            for(int idx=0;idx<(flow_based?flows:frames);idx++){
                strm << traces_ts[idx] << "," << traces_Val[i][idx] << " ";
            }
            strm << endl;
        }
    }
}

void SpatialPlot::DoubleClick_traces(int x, int y, int ts, float val)
{
    // .... override me ......
    qDebug() << __PRETTY_FUNCTION__ << ": Y" << QString::number(y) << "_X" << QString::number(x) << " TS " << ts << "val " << QString::number(val);

//    (void)x;
//    (void)y;
//    (void)ts;
//    (void)val;
}
void SpatialPlot::axisDoubleClick_traces(QCPAxis *axis, QCPAxis::SelectablePart part, QMouseEvent *event)
{
    (void)axis;
    (void)part;
    (void)event;
    qDebug() << __PRETTY_FUNCTION__ ;

}


void SpatialPlot::mouseDoubleClickEvent_traces(QMouseEvent *event)
{
    double x = _mTracePlot->xAxis->pixelToCoord(event->pos().x());
    double y = _mTracePlot->yAxis->pixelToCoord(event->pos().y());

    double x_lower=_mTracePlot->xAxis->range().lower;
    double x_upper=_mTracePlot->xAxis->range().upper;
    double y_lower=_mTracePlot->yAxis->range().lower;
    double y_upper=_mTracePlot->yAxis->range().upper;

    qDebug() << __PRETTY_FUNCTION__ << ": X" << x << "_Y" << y;

    if(x < x_lower){
        //handle y limits
        qDebug() << "handle y limits " << x;
        if(y < (y_lower + (y_upper - y_lower)/2)){
            bool ok;
            double rc = QInputDialog::getDouble(this,tr(""),tr("Y Min:"),y_lower,-1500,1500,1,&ok);
            Y_Lower_Override=ok?rc:NO_OVERRIDE;
        }else{
            bool ok;
            double rc = QInputDialog::getDouble(this,tr(""),tr("Y Max:"),y_upper,-1500,1500,1,&ok);
            Y_Upper_Override=ok?rc:NO_OVERRIDE;

        }
        render();
    }else if(y < y_lower){
        // handle x limits
        qDebug() << "handle x limits " << y;
        if(x < (x_lower + (x_upper - x_lower)/2)){
            bool ok;
            double rc = QInputDialog::getDouble(this,tr(""),tr("X Min:"),x_lower,-1500,1500,1,&ok);
            X_Lower_Override=ok?rc:NO_OVERRIDE;
        }else{
            bool ok;
            double rc = QInputDialog::getDouble(this,tr(""),tr("X Max:"),x_upper,-1500,1500,1,&ok);
            X_Upper_Override=ok?rc:NO_OVERRIDE;
        }
        render();
    }else{

        // figure out which frame we are looking at...
        int min_ts=0;
        for(int ts=0;ts<traces_len;ts++){
            if(std::abs(x - traces_ts[ts]) < std::abs(x - traces_ts[min_ts]))
                min_ts = ts;
        }

        int min_val_idx=0;
        for(int i=0;i<(MAX_NUM_TRACES);i++){
            if((traces[i].x >=0) && (traces[i].y > 0))
            {
                if(std::abs(y - traces_Val[i][min_ts]) < std::abs(y - traces_Val[min_val_idx][min_ts]))
                    min_val_idx = i;
            }
        }

        // we now have min_ts and min_val_idx....
        if(std::abs(y-traces_Val[min_val_idx][min_ts]) < (3.0*_mTraceRange/100.0)){
            //_mTracePlot->setToolTip(QString("Y%1_X%2: (%3,%4)").arg(traces[min_val_idx].y).arg(traces[min_val_idx].x).arg(traces_ts[min_ts]).arg(traces_Val[min_val_idx][min_ts]));
            DoubleClick_traces(traces[min_val_idx].x,traces[min_val_idx].y,min_ts,traces_Val[min_val_idx][min_ts]);
        }
    }
}

void SpatialPlot::SetTracesToolTip(int y, int x, float ts, float val)
{
    _mTracePlot->setToolTip(QString("Y%1_X%2: (%3,%4)").arg(y).arg(x).arg(ts).arg(val));

}

void SpatialPlot::showPointToolTip_traces(QMouseEvent *event)
{

    float x = _mTracePlot->xAxis->pixelToCoord(event->pos().x());
    float y = _mTracePlot->yAxis->pixelToCoord(event->pos().y());


    // figure out which frame we are looking at...
    int min_ts=0;
    for(int ts=0;ts<traces_len;ts++){
        if(std::abs(x - traces_ts[ts]) < std::abs(x - traces_ts[min_ts]))
            min_ts = ts;
    }

    int min_val_idx=0;
    for(int i=0;i<(MAX_NUM_TRACES);i++){
        if((traces[i].x >=0) && (traces[i].y > 0))
        {
            if(std::abs(y - traces_Val[i][min_ts]) < std::abs(y - traces_Val[min_val_idx][min_ts]))
                min_val_idx = i;
        }
    }

    // we now have min_ts and min_val_idx....
    if(std::abs(y-traces_Val[min_val_idx][min_ts]) < (3.0*_mTraceRange/100.0)){
        SetTracesToolTip(traces[min_val_idx].y,traces[min_val_idx].x,traces_ts[min_ts],traces_Val[min_val_idx][min_ts]);
    }
//    else{
//        _mTracePlot->setToolTip("");
//    }
}


void SpatialPlot::mouseDoubleClickEvent(QMouseEvent *event)
{

    if(event->x() >= (width()-scaleBarWidth)){
        // they double-clicked on the scale bar.
        if(event->y() < (height()/2)){
            // override the max value
            bool ok;
            double rc = QInputDialog::getDouble(this,tr(""),tr("Max:"),maxPixVal,-5000,20000,1,&ok);
            Spa_Upper_Override=ok?rc:NO_OVERRIDE;
        }else{
            // override the min value
            bool ok;
            double rc = QInputDialog::getDouble(this,tr(""),tr("Max:"),minPixVal,-5000,20000,1,&ok);
            Spa_Lower_Override=ok?rc:NO_OVERRIDE;
        }

    }else{
        int oobx=0;
        int ooby=0;
        int ax = GetAX(event->x(),width(),oobx);
        int ay = GetAY(event->y(),height(),ooby);
        int oob=oobx|ooby;
        bool ok=false;
        if(event->x() < borderWidth){
        	if(event->y() < borderHeight){
                bool ok;
                double rc = QInputDialog::getDouble(this,tr(""),tr("Y Max:"),endY,0,rows,1,&ok);
                if(ok)endY=rc;
        	}else if(event->y() > (height()-(2*borderHeight)) &&
        			(event->y() < (height()-(borderHeight)))){
                double rc = QInputDialog::getDouble(this,tr(""),tr("Y Min:"),startY,0,rows,1,&ok);
                if(ok)startY=rc;
        	}
        }else if(event->y() > (height() - borderHeight)){

        	if(event->x() > borderWidth && event->x() < (2*borderWidth)){
                double rc = QInputDialog::getDouble(this,tr(""),tr("X Min:"),startX,0,cols,1,&ok);
                if(ok)startX=rc;
        	}else if(event->x() > (width()-scaleBarWidth-borderWidth) && event->x() < (width()-scaleBarWidth)){
                double rc = QInputDialog::getDouble(this,tr(""),tr("X Max:"),endX,0,cols,1,&ok);
                if(ok)endX=rc;
        	}
        }
        DoubleClick(oob?-1:ax,oob?-1:ay);
        printf("%s: %d/%d  %d/%d\n",__FUNCTION__,event->x(),event->y(),ax,ay);
    }
    render();
    initialHints=0;
    fflush(stdout);
}
void SpatialPlot::mousePressEvent(QMouseEvent *event){
    LastPressX = event->x();
    LastPressY = event->y();
    printf("%s: %d/%d\n",__FUNCTION__,LastPressX,LastPressY);
    fflush(stdout);
}

void SpatialPlot::mouseReleaseEvent(QMouseEvent *event){
    printf("%s: %d/%d  last=%d/%d\n",__FUNCTION__,event->x(),event->y(),LastPressX,LastPressY);
    fflush(stdout);
    if((LastPressX == event->x()) &&
       (LastPressY == event->y())){
        // this was a click event..
        printf("%s: calling DoubleClick()\n",__FUNCTION__);
        mouseDoubleClickEvent(event);//->x(),event->y(),height(),width());
    }
}


//! [10]
void SpatialPlot::resizeEvent(QResizeEvent * /* event */)
{
//    render();
}
//! [10]

//! [11]
void SpatialPlot::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Plus:
        zoom(ZoomInFactor);
        break;
    case Qt::Key_Minus:
        zoom(ZoomOutFactor);
        break;
    case Qt::Key_Left:
        {
            //scroll(-ScrollStep, 0);
            int trc=traces_curSelection;
            if(traces[trc].x > 0){
                traces[trc].x--;
                render();
            }
        }
        break;
    case Qt::Key_Right:
        {
            //scroll(+ScrollStep, 0);
            int trc=traces_curSelection;
            if((traces[trc].x+1) < cols){
                traces[trc].x++;
                render();
            }
        }
        break;
    case Qt::Key_Down:
        {
            //scroll(0, -ScrollStep);
            int trc=traces_curSelection;
            if((traces[trc].y) > 0){
                traces[trc].y--;
                render();
            }

        }
        break;
    case Qt::Key_Up:
        {
            //scroll(0, +ScrollStep);
            int trc=traces_curSelection;
            if((traces[trc].y+1) < rows){
                traces[trc].y++;
                render();
            }
        }
        break;
    default:
        QWidget::keyPressEvent(event);
    }
}
//! [11]

#ifndef QT_NO_WHEELEVENT
//! [12]
void SpatialPlot::wheelEvent(QWheelEvent *event)
{
    int numDegrees = event->delta() / 8;
    double numSteps = numDegrees / 15.0f;
    double zf = pow(ZoomInFactor, numSteps);

    int oob=0;
    int x=event->pos().x();
    int y=event->pos().y();
    int ay=GetAY(y,height(),oob);
    int ax=GetAX(x,width(),oob);
    if(!oob){
        double WeightX = ((double)(ax - startX)) / ((double)(endX - startX));
        double WeightY = ((double)(ay - startY)) / ((double)(endY - startY));
        printf("%s: %lf %d/%d %d/%d %lf/%lf\n",__FUNCTION__,zf,y,x,ay,ax,WeightY,WeightX);
        fflush(stdout);
        zoom(zf,WeightX,WeightY);
    }
}
//! [12]
#endif

////! [13]
//void SpatialPlot::mousePressEvent(QMouseEvent *event)
//{
//    if (event->button() == Qt::LeftButton)
//        lastDragPos = event->pos();
//}
////! [13]

//! [14]
void SpatialPlot::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton && lastDragPos.x()) {
        //int width = endX-startX;
        QPoint diff = event->pos() - lastDragPos;
        int diffX = (diff.x() * (endX-startX)/width()) ;
        int diffY = (diff.y() * (endY-startY)/height()) ;
        diffY = -diffY; //it's flipped
        printf("%s: %d/%d --> %d/%d\n",__FUNCTION__,event->y(),event->x(),lastDragPos.y(),lastDragPos.x());
        printf("%s: %d/%d/%d/%d --> %d/%d/%d/%d\n",__FUNCTION__,startY,startX,endY,endX,startY-diffY,startX-diffX,endY-diffY,endX-diffX);
        fflush(stdout);
        if(diffX > startX)
            diffX = startX;
        if(diffY > startY)
            diffY = startY;
        if((endX-diffX) > cols)
            diffX = endX-cols;
        if((endY-diffY) > rows)
            diffY = endY-rows;
        startX -= diffX;
        startY -= diffY;
        endX -= diffX;
        endY -= diffY;
        if(diffX || diffY){
            lastDragPos = event->pos();
            initialHints=0;
        }
        render();
    }
    else{
        lastDragPos = event->pos();
    }
}
//! [14]

#if 0
//! [15]
void SpatialPlot::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        pixmapOffset += event->pos() - lastDragPos;
        lastDragPos = QPoint();

        int deltaX = (width() - pixmap.width()) / 2 - pixmapOffset.x();
        int deltaY = (height() - pixmap.height()) / 2 - pixmapOffset.y();
        scroll(deltaX, deltaY);
    }
}
//! [15]
#endif
//! [16]
void SpatialPlot::updatePixmap(const QImage &image)
{
    pixmap = QPixmap::fromImage(image);
    UpdateTracePlot();
    update();

}
//! [16]

//! [17]
void SpatialPlot::zoom(double zoomFactor, double XWeight, double YWeight)
{
    int diffX = (int)((double)(endX-startX) * (1.0-zoomFactor))/2;
    int diffY = (int)((double)(endY-startY) * (1.0-zoomFactor))/2;
    if(diffX >= 0 && diffX < 4)
        diffX=4;
    if(diffY >= 0 && diffY < 4)
        diffY=4;

    if(diffX <= 0 && diffX > -4)
        diffX=-4;
    if(diffY <= 0 && diffY > -4)
        diffY=-4;

    int diffXS = (int)((double)diffX * XWeight);
    int diffXE = diffX-diffXS;
    int diffYS = (int)((double)diffY * YWeight);
    int diffYE = diffY-diffYS;

    qDebug() << __FUNCTION__ << ": (" << startY << "/" << startX << ") (" << endY << "/" << endX << ") zf:" << zoomFactor << " diffS(" << diffYS << "/" << diffXS << ") diffE(" << diffYE << "/" << diffXE << ")";

    startX += diffXS;
    startY += diffYS;
    endX -= diffXE;
    endY -= diffYE;
    if(startX < 0)
        startX=0;
    if(startY<0)
        startY=0;
    if(endX > cols)
        endX=cols;
    if(endY > rows)
        endY=rows;
    initialHints=0;
    render();
}
//! [17]

//! [18]
void SpatialPlot::scroll(int deltaX, int deltaY)
{
    int diffX = (int)((double)(endX-startX) * (deltaX/width()));
    int diffY = (int)((double)(endY-startY) * (deltaY/height()));
    startX += diffX;
    startY -= diffY;
    endX += diffX;
    endY -= diffY;
    initialHints=0;
    update();
}
//! [18]

void SpatialPlot::doConvertInt()
{
    doConvert(loading);
//    loading=0;
}

void SpatialPlot::doConvert(int &loading){(void)loading;}


void SpatialPlot::copyData()
{
    _fname = fname;

}

//! [0]
SpatialPlotThread::SpatialPlotThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;

}
void SpatialPlotThread::SetParent(SpatialPlot *_parent)
{
    mParent = _parent;
}

//! [0]

//! [1]
SpatialPlotThread::~SpatialPlotThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}
//! [1]

void SpatialPlot::copyLastCoord()
{
    lastStartX = startX;
    lastStartY = startY;
    lastEndX   = endX;
    lastEndY   = endY;
}

//! [2]
void SpatialPlotThread::render()
{
    mParent->copyLastCoord();
    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}
//! [2]

void SpatialPlotThread::Abort()
{
    abort=true;
    mutex.unlock();
}

void SpatialPlotThread::run()
{
    forever {
        mutex.lock();
        if(abort)
            return; // application is closed
        mParent->copyData();
        mutex.unlock();

        mParent->doConvertInt();

        QImage *image = mParent->doRender();
        if(image){
            emit renderedImage(*image);
            free(image);
        }

        mutex.lock();
        if (!restart)
            condition.wait(&mutex);
        restart = false;
        mutex.unlock();
    }
}

// get data coords from screen coords
int SpatialPlot::GetAY(int y, int Height, int & ooby)
{
    double bh = borderHeight;
    double rh = (double)(endY-startY);
    double screenh = (double)(Height - bh);
    int ay = endY - 1 - (int)(((double)y) / screenh * rh);
    if((y >= screenh))
        ooby=1;
    if(ay >= rows || ay < 0)
        ooby = 1;
    return ay;
}

// get data coords from screen coords
int SpatialPlot::GetAX(int x, int Width, int & oobx)
{
    int bw = borderWidth+scaleBarWidth+2;
    double rw = (double)(endX-startX);
    double screenw = (double)(Width-bw);
    int ax = startX + (int)((double)(x-borderWidth) * rw/screenw);
    if(x < borderWidth || (x > (Width-(scaleBarWidth+2))))
        oobx=1;
    if(ax >= cols || ax < 0)
        oobx=1;
    return ax;
}

// get screen coords from data coords
int SpatialPlot::GetY(int ay, int Height, int & ooby)
{
    double rh = (double)(endY-startY);
    double screenh = (double)(Height - borderHeight);
    int y = (int)((double)(endY-ay-1) * screenh/rh);
    if(y < 0)
        ooby=1;
    if(y >= Height)
        ooby=1;

    return y;
}

// get screen coords from data coords
int SpatialPlot::GetX(int ax, int Width, int & oobx)
{
    int bw = borderWidth+scaleBarWidth+2;
    double rw = (double)(endX-startX);
    double screenw = (double)(Width-bw);
    ax -= startX;
    int x = borderWidth + (int)((double)ax * screenw/rw);
    if(x < 0)
        oobx=1;
    if(x >= Width)
        oobx=1;
    return x;
}


QImage *SpatialPlot::doRender()
{
    int Height = height();
    int Width  = width();
    QImage *image = new QImage(Width,Height, QImage::Format_RGB32);

    if(startX < 0)
        startX=0;
    if(startX > cols)
        startX = cols;
    if(startY<0)
        startY=0;
    if(startY > rows)
        startY = rows;
//        printf("rendering from %d/%d/%d to %d/%d --> %d/%d frame %d\n",
//               rows,cols,frames,startX,startY,endX,endY, frame);

    float minVal = 16384;
    float maxVal = -16384;
    {
        // get the max and min values first...
        for (int y = 0; y < Height; y++) {
            int ooby=0;
            int ay = GetAY(y,Height,ooby);
            for(int x=0;x<Width;x++){
                int oobx=0;
                int ax = GetAX(x,Width,oobx);

                if(ooby || oobx){
                }else{

                    float val = Get_Data(flow_based?flow:frame,ay,ax);
                    if((val < minVal) && (!ignore_pinned || (val >= 4))){
                        minVal = val;
                    }
                    if((val > maxVal) && (val <= 16380)){
                        maxVal = val;
                    }
                }
            }
        }

        maxVal++;
        if(maxVal <= minVal)
            maxVal = minVal+10;
    //    if(minVal > 0)
    //        minVal-=1;
    }

    if(Spa_Lower_Override != NO_OVERRIDE)
        minVal = Spa_Lower_Override;
    if(Spa_Upper_Override != NO_OVERRIDE)
        maxVal = Spa_Upper_Override;


    maxPixVal=maxVal;
    minPixVal=minVal;

    float range = maxVal-minVal;
    float histBlockSize = range/sizeof(sizeof(histData)/sizeof(histData[0]));
    if(histBlockSize < 1)
    	histBlockSize = 1;
    memset(histData,0, (sizeof(histData)));
//    if(range < 1)
//        range=1; // don't divide by zero!
//    printf("range = %d - %d --> %d\n",maxVal,minVal,range);

    for (int y = 0; y < Height; y++) {
        uint *scanLine =
                reinterpret_cast<uint *>(image->scanLine(y));
        int ooby=0;
        int ay = GetAY(y,Height,ooby);
        for(int x=0;x<Width;x++){
            int oobx=0;
            int ax = GetAX(x,Width,oobx);

            if(ooby || oobx){
                if(x > (Width-scaleBarWidth+2)){
                    int val = (Height-y-1) * ColormapSize / Height;
                    scanLine[x] = colormap[val % ColormapSize ];
                }else{
                    scanLine[x] = 0;//colormap[0];
                }
            }else{

                float val = Get_Data(flow_based?flow:frame,ay,ax);
                histData[((uint32_t)val/(uint32_t)histBlockSize) % (sizeof(histData)/sizeof(histData[0]))]++;
                if(val > maxVal)
                    val = maxVal;
                if(val < minVal)
                    val = minVal;
//                if(ignore_pinned && ((val >= 16380) || (val <= 4)))
//                    scanLine[x]=colormap[0];
//                else{
                    val -= minVal;
                    val *= ColormapSize;
                    val /= range;
                    uint cmap = colormap[(int)val % ColormapSize];
                    scanLine[x] = cmap;
//                }
            }
        }
    }

    paintXMarks(image);
    UpdateTraceData();

    return image;
}

void SpatialPlot::paintXMarks(QImage *image)
{
    // put the X's on the selected well's
    if(rows && cols){
        for(int ts=0;ts<MAX_NUM_TRACES;ts++){
            if(traces[ts].x < 0 || traces[ts].y < 0)
                continue;

            int pix=5;
            int oobx=0;
            int ooby=0;
            int x=(GetX(traces[ts].x,width(),oobx) + GetX(traces[ts].x+1,width(),oobx))/2;
            int y=(GetY(traces[ts].y,height(),ooby) + GetY(traces[ts].y-1,height(),ooby))/2;
            if(oobx==0 && ooby==0){
                uint xcolorIdx= ts*ColormapSize/MAX_NUM_TRACES;
                uint xcolor=colormap[xcolorIdx % ColormapSize];
                // make an X at x/y
                for(int marky=(y-pix);marky<(y+pix);marky++){
                    if(marky < 0 || marky >= height())
                        continue;
                    uint *scanLine =
                            reinterpret_cast<uint *>(image->scanLine(marky));
                    if(marky != y){
                        scanLine[x]=xcolor;
                    }else{
                        for(int markx=(x-pix);markx<(x+pix);markx++){
                            if(markx < 0 || markx >= width())
                                continue;
                            scanLine[markx]=xcolor;
                        }
                    }
                }

            }
        }
    }
}

float SpatialPlot::Get_Data(int idx, int y, int x)
{
    (void)idx;
    (void)y;
    (void)x;
    return 0;
}

//! [10]
uint SpatialPlot::rgbFromWaveLength(double wave)
{
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;

    if (wave >= 380.0 && wave <= 440.0) {
        r = -1.0 * (wave - 440.0) / (440.0 - 380.0);
        b = 1.0;
    } else if (wave >= 440.0 && wave <= 490.0) {
        g = (wave - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (wave >= 490.0 && wave <= 510.0) {
        g = 1.0;
        b = -1.0 * (wave - 510.0) / (510.0 - 490.0);
    } else if (wave >= 510.0 && wave <= 580.0) {
        r = (wave - 510.0) / (580.0 - 510.0);
        g = 1.0;
    } else if (wave >= 580.0 && wave <= 645.0) {
        r = 1.0;
        g = -1.0 * (wave - 645.0) / (645.0 - 580.0);
    } else if (wave >= 645.0 && wave <= 780.0) {
        r = 1.0;
    }

    double s = 1.0;
    if (wave > 700.0)
        s = 0.3 + 0.7 * (780.0 - wave) / (780.0 - 700.0);
    else if (wave <  420.0)
        s = 0.3 + 0.7 * (wave - 380.0) / (420.0 - 380.0);

    r = pow(r * s, 0.8);
    g = pow(g * s, 0.8);
    b = pow(b * s, 0.8);
    return qRgb(int(r * 255), int(g * 255), int(b * 255));
}
