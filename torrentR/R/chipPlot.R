chipPlot <- function(zCol,zRow,zVal,minCol=0,minRow=0,maxCol=NA,maxRow=NA,zlim=NA,nColBin=100,nRowBin=100,minBin=5,header="",histLim=NA,doPlot=TRUE,doInterpolate=FALSE,cex.header=1,summaryFunction=c("mean","median"),color=rgb(rep(0,256),seq(0,1,length=256),seq(1,0,length=256))) {

  summaryFunction <- match.arg(summaryFunction)

  # Collapse to the per-bin median
  binned <- bin2D(zCol,zRow,zVal,minX=minCol,minY=minRow,maxX=maxCol,maxY=maxRow,nBinX=nColBin,nBinY=nRowBin,minBin=minBin)
  if(summaryFunction == "median") {
    binned$z.summary <- unlist(lapply(binned$z,median,na.rm=TRUE))
  } else {
    binned$z.summary <- unlist(lapply(binned$z,mean,na.rm=TRUE))
  }

  # Reshape results to a matrix that image() can plot
  z.image <- formImageMatrix(binned$x,binned$y,binned$z.summary,nColBin,nRowBin)

  # Do the plot
  if(doPlot)
    imageWithHist(z.image,zlim=zlim,header=header,histLim=histLim,cex.header=cex.header,col=color)

  ret <- list(
    nColBin   = nColBin,
    nRowBin   = nRowBin,
    binnedCol = binned$x,
    binnedRow = binned$y,
    binnedVal = binned$z.summary
  )

  # Interpolate summarized values back out chip-wide and return
  if(doInterpolate) {
    interpObj <- list(
      x = maxCol/nColBin*((1:nColBin)-0.5),
      y = maxRow/nRowBin*((1:nRowBin)-0.5),
      z = z.image
    )
    xSlice <- 1:maxCol
    ySlice <- 1:maxRow
    zSlice <- matrix(interp.surface(interpObj,expand.grid(xSlice,ySlice)),length(xSlice),length(ySlice))
    ret$interpVal <- zSlice
  }

  return(ret)
}
