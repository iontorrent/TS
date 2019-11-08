/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MicroscopeSpatial.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>

using namespace std;


MicroscopeSpatial::MicroscopeSpatial(QWidget *parent): SpatialPlot(parent)
{
//     imageLabel.setBackgroundRole(QPalette::Base);
//     imageLabel.setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//     imageLabel.setScaledContents(true);

//     scrollArea.setBackgroundRole(QPalette::Dark);
//     //scrollArea.setWidget(imageLabel);
//     scrollArea.setVisible(false);
//     //setCentralWidget(scrollArea);

}


void MicroscopeSpatial::setSlider(int per)
{
    (void)per;
//    QualityLimit = per;
    render();
}

void MicroscopeSpatial::SetDualFileName(QString fn1, QString fn2)
{
    for(int i=0;i<MAX_MS_FN;i++){
        if(fileNameImg[i].length() == 0){
            fileNameImg[i] = fn1;
            fileNameCsv[i] = fn2;
            loaded[i]=0;
            break;
        }
    }
}


void MicroscopeSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    render();
}

void MicroscopeSpatial::doConvert(int &loading)
{
    for(int fnIdx=0;fnIdx<MAX_MS_FN;fnIdx++){

        if(rows && cols /*&& !loading*/ && fileNameImg[fnIdx] != "" && fileNameCsv[fnIdx] != "" && !loaded[fnIdx]){
            loading=1;
            loaded[fnIdx]=true;

            // load a new dat file
            printf("triggered %s %s\n",fileNameImg[fnIdx].toLatin1().data(),fileNameImg[fnIdx].toLatin1().data());
    //        last_fname = _fname;
            //if(_fname.contains(".csv"))
            {
                int len = rows*cols*sizeof(microscopeOut[0]);
                if(microscopeOut == NULL){
                    microscopeOut=(float *)malloc(len);
                    memset(microscopeOut,0,len);
                }
                QString fn2 = fileNameCsv[fnIdx];
                char *fn = fn2.toLatin1().data();
                int bestDistX=0;
                int bestDistY=0;
                int startX_x=-1;
                int startY_y=-1;

                {
                  //Note, fillMask is optional argument w/ default of true.
                  FILE *in = fopen ( fn, "r" );
                  if(in){

                      char line[1024];
        //              size_t lineLen=0;
                      //int ch=0;
                      fgets(line,sizeof(line),in);
                      printf("%s: %s\n",__FUNCTION__,line);
                      while(fgets(line,sizeof(line),in)){
                          printf("%s: %s\n",__FUNCTION__,line);
                          // parse one pixel value
                          int col=0;
                          int row=0;
                          int x=0;
                          int y=0;
                          float val=0;
                          int numScanned = fscanf(in,"%d,%d,%d,%d,%f",&col,&row,&x,&y,&val);
                          if(numScanned == 5){
                              if(col < cols && row < rows){
                                  microscopeOut[row*cols+col] = val;
                                  if(col < pixelXStart[fnIdx] || pixelXStart[fnIdx] < 0){
                                      pixelXStart[fnIdx] = col;
                                      startX_x = x;
                                  }
                                  if(row < pixelYStart[fnIdx] || pixelYStart[fnIdx] < 0){
                                      pixelYStart[fnIdx]=row;
                                      startY_y=y;
                                  }
                                  int distX = col - pixelXStart[fnIdx];
                                  int distY = row - pixelYStart[fnIdx];

                                  if(distX && distX > bestDistX){
                                      bestDistX = distX;
                                      pixelWidth[fnIdx] = (float)(x-startX_x)/(float)distX;
                                  }
                                  if(distY && distY > bestDistY){
                                      bestDistY = distY;
                                      pixelHeight[fnIdx] = (float)(startY_y-y)/(float)distY;
                                  }
                              }
                          }else{
                              printf("format problems %d\n",numScanned);
                          }
                      }
                      fclose ( in );
                  }
                }
            }/////else if(_fname.contains(".jpg") || _fname.contains(".png"))
            {
                QImageReader reader(fileNameImg[fnIdx]);
                //reader.setAutoTransform(true);
                QimgSrc[fnIdx] = reader.read();
    //            if (newImage.isNull()) {
    //                printf("failed to load jpg file\n");
    //            }else{
    //                pixmap = QPixmap::fromImage(newImage);
                    //imageLabel->setPixmap(QPixmap::fromImage(image));
                    //scaleFactor = 1.0;

    //            }
            }
            traces_len=2;
            for(int i=0;i<traces_len;i++)
                traces_ts[i]=i;

            printf("\ndone loading microscope File %s\n\n",last_fname.toLatin1().data());
            fflush(stdout);
            loading=0;
        }
    }
}

uint MicroscopeSpatial::GetPixVal(int y, int x)
{
    int rc = 0;

    for(int fnIdx=0;fnIdx<MAX_MS_FN;fnIdx++){
        if(QimgSrc[fnIdx].height() > 0  && QimgSrc[fnIdx].width() > 0){

            int ry = y;
            int rx = x;
            int rheight=0;
            int rwidth=0;

            if(pixelWidth[fnIdx] >= 0 && pixelHeight[fnIdx] >= 0 && pixelXStart[fnIdx] >= 0 && pixelYStart[fnIdx] >= 0){
                rheight = (int)((float)(endY-startY+1) * pixelHeight[fnIdx]); // +1 to see the full edge pixels
                rwidth  = (int)((float)(endX-startX+1) * pixelWidth[fnIdx]); // +1 to see the full edge pixels

                ry = (int)((float)rheight * (float)y / (float)height());
                rx = (int)((float)rwidth * (float)x / (float)width());

                ry -= (int)((float)pixelYStart[fnIdx] * pixelHeight[fnIdx]);
                rx -= (int)((float)pixelXStart[fnIdx] * pixelWidth[fnIdx]);

            }
//            else{
//                ry = y*QimgSrc[fnIdx].height()/height();
//                rx = x*QimgSrc[fnIdx].width()/width();
//            }
            if(rx >= 0 && rx < QimgSrc[fnIdx].width() &&
                    ry >= 0 && ry < QimgSrc[fnIdx].height()){
                uint *scanLine =
                        reinterpret_cast<uint *>(QimgSrc[fnIdx].scanLine(ry));
                rc = scanLine[rx];
            }
        }
    }
    return rc;
}

QImage *MicroscopeSpatial::doRender()
{
    int Height = height();
    int Width  = width();
    QImage *image = new QImage(Width,Height, QImage::Format_RGB32);

    // we need to scale the image the same as the rest of the data
    if(startX < 0)
        startX=0;
    if(startX > cols)
        startX = cols;
    if(startY<0)
        startY=0;
    if(startY > rows)
        startY = rows;

    float screenh = (float)(Height - borderHeight);

    // write all 0's to the image..
    for(int y=0;y<Height;y++){
        uint *scanLineDst = reinterpret_cast<uint *>(image->scanLine(y));
        for(int x=0;x<Width;x++){
            scanLineDst[x]=0;
        }
    }

    for(int fnIdx=0;fnIdx<MAX_MS_FN;fnIdx++){

        float rheight = ((float)(endY-startY+1) * pixelHeight[fnIdx]); // +1 to see the full edge pixels
        float rwidth  = ((float)(endX-startX+1) * pixelWidth[fnIdx]); // +1 to see the full edge pixels

        float rx_Start = (pixelXStart[fnIdx]-startX) * pixelWidth[fnIdx];
        float ry_Start = (pixelYStart[fnIdx]-startY) * pixelHeight[fnIdx];


        for(int y=0;y<Height;y++){
            if(y < (Height-borderHeight-1)){
                int ry = y;
                if(pixelWidth[fnIdx] >= 0 && pixelHeight[fnIdx] >= 0 && pixelXStart[fnIdx] >= 0 && pixelYStart[fnIdx] >= 0){
                    ry = (int)((float)rheight * (float)(screenh-y) / (float)screenh);
                    ry -= ry_Start;
                }

                uint *scanLineDst = reinterpret_cast<uint *>(image->scanLine(y));
                uint *scanLineSrc = NULL;
                if(ry >= 0 && ry < QimgSrc[fnIdx].height()){
                        scanLineSrc = reinterpret_cast<uint *>(QimgSrc[fnIdx].scanLine(ry));
                }
                for(int x=0;x<Width;x++){
                    if(x<(Width-(scaleBarWidth+2)) && x > borderWidth)
                    {
                        int rx = x-borderWidth;
                        if(pixelWidth[fnIdx] >= 0 && pixelHeight[fnIdx] >= 0 &&
                                pixelXStart[fnIdx] >= 0 && pixelYStart[fnIdx] >= 0){
                            rx = (int)((float)rwidth * (float)rx / (float)width());
                            rx -= rx_Start;
                        }
                        if(rx >= 0 && rx < QimgSrc[fnIdx].width() && scanLineSrc)
                            scanLineDst[x]=scanLineSrc[rx];
                    }
                }
            }
        }
    }

    paintXMarks(image);

    return image;
}

//void MicroscopeSpatial::paintEvent(QPaintEvent * /* event */)
//{
//    if(!firstShow ||
//            (lastStartX != startX) ||
//            (lastStartY != startY) ||
//            (lastEndX   != endX) ||
//            (lastEndY   != endY) ||
//            //(lastcurDisplayFlow != curDisplayFlow) ||
//            (GblTracesVer != LocalTracesVer)){
//        thread.render();
//    }
//    firstShow=1;
//    QPainter painter(this);
//    painter.fillRect(rect(), Qt::black);
//    else if (pixmap.isNull()) {
//        painter.setPen(Qt::white);
//        painter.drawText(rect(), Qt::AlignCenter, tr("Rendering initial image, please wait..."));
//        return;
//    }
//    painter.drawPixmap(QPoint(0,0)/*pixmapOffset*/, pixmap);
//    imageLabel->setPixmap(QPixmap::fromImage(image));
//}

float MicroscopeSpatial::Get_Data(int frame, int y, int x)
{
    (void)frame;
    float rc = 0;

    if(microscopeOut && y < rows && x < cols){
        rc = microscopeOut[y*cols+x];
    }
    return rc;
}






