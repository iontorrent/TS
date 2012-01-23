tfFlowPlot <- function(
  dataDir,
  tfKeySeq,
  plotDir=NA,
  setName=NA,
  minTfNumber=5000,
  flowRange=1:50,
  minBin=6,
  plotType=c("png","bitmap","pdf")
) {

  plotType <- match.arg(plotType)

  # Set random seed for reproducible sampling
  set.seed(0)

  # If setName wasn't specified use the base name of the data directory
  if(is.na(setName))
    setName <- basename(dataDir)
          
  # Add trailing file separators if not already present
  if(.Platform$file.sep != rev(unlist(strsplit(dataDir,"")))[1])
    dataDir <- sprintf("%s%s",dataDir,.Platform$file.sep)
  if(plotType != "none") {
    if(.Platform$file.sep != rev(unlist(strsplit(plotDir,"")))[1])
      plotDir <- sprintf("%s%s",plotDir,.Platform$file.sep)
  }

  tfSet <- readTfConf(dataDir)
  if(is.null(tfSet)) {
    warning(sprintf("Unable to read TF config for %s\n",dataDir))
  } else {
    tfStats <- readTfStats(dataDir)
    if(is.null(tfStats)) {
      warning(sprintf("Unable to read TF tracking data for %s\n",dataDir))
    } else if(length(tfStats[[1]])==0) {
      warning(sprintf("No TFs found for %s\n",dataDir))
    } else {

      tfName <- sort(names(which(table(tfStats$tfSeq) > minTfNumber)))
      nTfTypes <- length(tfName)
  
      if(nTfTypes==0) {
        warning(sprintf("Insufficient TFs for %s\n  Need %d, most frequent was %d.  Skipping...\n",dataDir,minTfNumber,max(table(tfStats$tfSeq))))
      } else {
        bfMaskFile <- paste(dataDir,"bfmask.bin",sep="")
        wellFile <- paste(dataDir,"1.wells",sep="")
        bf <- readBeadFindMask(bfMaskFile)
        pctLoaded  <- mean(bf$maskBead)
        for(tfCounter in 1:nTfTypes) {
          tfMask <- tfStats$tfSeq==tfName[tfCounter]
          nTF <- sum(tfMask)
          nBin   <- floor(sqrt(nTF/(6*minBin)))
          x <- readWells(wellFile,col=tfStats$col[tfMask],row=tfStats$row[tfMask],flow=flowRange)
          xNorm <- normalizeIonogram(x$signal,tfKeySeq,x$flowOrder)$norm
          if(sum(temp <- tfSet$tf==tfName[tfCounter] & tfSet$keySeq==tfKeySeq) == 1) {
            trueSeq <- paste(tfSet$keySeq[temp],tfSet$tfSeq[temp],sep="")
            trueFlow <- seqToFlow(trueSeq,x$flowOrder)
          } else {
            trueSeq <- NULL
            trueFlow <- NULL
          }
          if(plotType == "pdf")
            pdf(file=sprintf("%stf.%s.pdf",plotDir,tfName[tfCounter]),height=8,width=6)
          for(flow in flowRange) {
            flowIndex <- which(x$flow==flow)
            zVal <- xNorm[,flowIndex]
            
            binned <- bin2D(x$col,x$row,zVal,maxX=x$nCol,maxY=x$nRow,nBinX=nBin,nBinY=nBin)
            binSize <- unlist(lapply(binned$z,length))
            binned$plotVal <- unlist(lapply(binned$z,median,na.rm=TRUE))
            binned$plotVal[binSize < minBin] <- NA
            binned$imageMatrix <- formImageMatrix(binned$x,binned$y,binned$plotVal,nBin,nBin)
            lim <- quantile(binned$imageMatrix,probs=c(0.02,0.98),na.rm=TRUE)
            if(plotType != "pdf")
              plotHelper(sprintf("%stf.%s.flow%03d.png",plotDir,tfName[tfCounter],flow),height=800,width=600)
            imageWithHist(binned$imageMatrix,zlim=lim,histLim=lim)
            if(!is.null(trueFlow)) {
              flowString <- sprintf("; %d-mer %s",trueFlow[flow],x$flowBase[flowIndex])
              if(flow > 1)
                flowString <- sprintf("%s, prev=%s",flowString,paste(trueFlow[(max(flow-4,1)):(flow-1)],collapse="-"))
              if(flow < length(trueFlow))
                flowString <- sprintf("%s, next=%s",flowString,paste(trueFlow[(flow+1):min(flow+4,length(trueFlow))],collapse="-"))
            } else {
              flowString <- ""
            }
            mtext(sprintf("%s; %d reads for %s\nflow %03d%s",setName,nTF,tfName[tfCounter],flow,flowString),outer=TRUE,line=-0.5,cex=1.2)
            if(plotType != "pdf")
              dev.off()
          }
          if(plotType == "pdf")
            dev.off()
        }
      }
    }
  }
}
