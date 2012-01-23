flowFrameTracePlot <- function(
  myCol,
  myRow,
  myVal,
  baseName,
  frameRange,
  header,
  maxCol,
  maxRow,
  nColBin,
  nRowBin,
  minBin=100,
  flowPlotHeight=600,
  flowPlotWidth=450,
  plotType=c("png","bitmap","pdf","none"),
  doInterpolate=FALSE,
  wellSignal=NA
) {
  plotType <- match.arg(plotType)

  # Make a plot of the data for each flow
  lim <- quantile(myVal,probs=c(0.02,0.995),na.rm=TRUE)
  if(plotType == "pdf")
    pdf(file=sprintf("%s.flow.pdf",baseName),height=flowPlotHeight/100,width=flowPlotWidth/100)
  if(doInterpolate)
    interpVal  <- matrix(NA,nrow=maxCol*maxRow,ncol=length(frameRange))
  binnedVal <- numeric()
  for(frameIndex in 1:length(frameRange)) {
    frame <- frameRange[frameIndex]
    if(is.element(plotType,c("png","bitmap")))
      plotHelper(sprintf("%s.flow.frame%03d.png",baseName,frame-1),height=flowPlotHeight,width=flowPlotWidth)
    thisHeader <- sprintf("%s; %d wells, frame %03d",header,nrow(myVal),frame-1)
    doPlot <- (plotType != "none")
    empty <- chipPlot(
      myCol,
      myRow,
      myVal[,frame],
      maxCol=maxCol,
      maxRow=maxRow,
      nColBin=nColBin,
      nRowBin=nRowBin,
      zlim=lim,
      histLim=lim,
      header=thisHeader,
      cex.header=0.85,
      minBin=minBin,
      doInterpolate=doInterpolate,
      doPlot=doPlot
    )
    if(doInterpolate)
      interpVal[,frameIndex] <- empty$interpVal
    if(is.element(plotType,c("png","bitmap")))
      dev.off()
    if(length(binnedVal)==0) {
      binnedCol <- empty$binnedCol
      binnedRow <- empty$binnedRow
      binnedVal <- matrix(NA,nrow=length(binnedCol),ncol=length(frameRange))
      binnedVal[,1] <- empty$binnedVal
    } else {
      binnedVal[,frameIndex] <- empty$binnedVal
    }
  }
  if(plotType == "pdf")
    dev.off()
  if(is.element(plotType,c("png","bitmap"))) {
    command <- sprintf("convert -delay 10 -loop 100 %s.flow.frame*.png %s.flow.gif",baseName,baseName)
    system(command)
    command <- sprintf("rm -f %s.flow.frame*.png",baseName)
    system(command)
  }

  # Make a plot of the estimated well signal, if provided
  if( (!any(is.na(wellSignal))) & is.element(plotType,c("png","bitmap","pdf")) ) {
    lim <- quantile(wellSignal,probs=c(0.02,0.99),na.rm=TRUE)
    if(is.element(plotType,c("png","bitmap"))) {
      plotHelper(sprintf("%s.wellSig.png",baseName),height=flowPlotHeight,width=flowPlotWidth)
    } else {
      pdf(file=sprintf("%s.wellSig.pdf",baseName),height=flowPlotHeight/100,width=flowPlotWidth/100)
    }
    thisHeader <- sprintf("%s; %d wells, well signal",header,nrow(myVal),frame-1)
    empty <- chipPlot(
      myCol,
      myRow,
      wellSignal,
      maxCol=maxCol,
      maxRow=maxRow,
      nColBin=nColBin,
      nRowBin=nRowBin,
      zlim=lim,
      histLim=lim,
      header=thisHeader,
      cex.header=0.85,
      minBin=minBin,
      doInterpolate=FALSE,
      doPlot=TRUE
    )
    dev.off()
  }

  # Make a plot of the per-region traces
  if(plotType == "pdf")
    pdf(file=sprintf("%s.trace.pdf",baseName),height=flowPlotWidth/100,width=flowPlotWidth/100)
  if(is.element(plotType,c("png","bitmap")))
    plotHelper(sprintf("%s.trace.png",baseName),height=flowPlotWidth,width=flowPlotWidth)
  nColor <- 256
  myPalette <- rgb(rep(0,nColor),seq(0,1,length=nColor),seq(1,0,length=nColor))
  traceIndex <- binnedRow*nColBin+binnedCol
  colorIndex <- 1+floor(nColor*traceIndex/(max(traceIndex)+1))
  myColor <- myPalette[colorIndex]
  ylim <- apply(apply(binnedVal,2,quantile,probs=c(0.02,0.98),na.rm=TRUE),1,quantile,probs=c(0.02,0.98),na.rm=TRUE)[c(1,4)]
  plot(range(frameRange),ylim,xlab="Frame",ylab="Binned intensity",type="n")
  for(i in sample(1:nrow(binnedVal))) {
    lines(frameRange,binnedVal[i,],lwd=0.3,col=myColor[i])
  }
  title(sprintf("%s, %d binned traces",header,nrow(binnedVal)))
  if(plotType != "none")
    dev.off()

  if(doInterpolate)
    return(interpVal)
}
