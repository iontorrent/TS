tfOneMerSNR <- function(
  dataDir,
  tfKeySeq,
  snrLim=c(0,12),
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
          tfSetMask <- tfSet$tf==tfName[tfCounter] & tfSet$keySeq==tfKeySeq
          if(sum(tfSetMask) == 0) {
            warning(sprintf("No TF entries for %s in %s, skipping...\n",tfName[tfCounter],dataDir))
          } else if(sum(tfSetMask) > 1) {
            warning(sprintf("Multiple TF entries for %s in %s, skipping...\n",tfName[tfCounter],dataDir))
          } else {
            trueSeq <- paste(tfSet$keySeq[tfSetMask],tfSet$tfSeq[tfSetMask],sep="")
            flowOrder <- paste(strsplit(x$flowOrder,"")[[1]][flowRange],collapse="")
            snrFit <- snrStats(trueSeq,flowOrder,tfKeySeq,x$signal)
            # Median Signal
            plotHelper(sprintf("%ssnr.med.%s.png",plotDir,tfName[tfCounter]),height=400,width=800)
            header <- sprintf("%s, %s\n(CF,IE,DR)=(%0.3f, %0.3f, %0.4f)",setName,tfName[tfCounter],snrFit$cf,snrFit$ie,snrFit$dr)
            snrPlotMedSig(snrFit,main=header)
            dev.off()
            # SD Signal
            plotHelper(sprintf("%ssnr.sd.%s.png",plotDir,tfName[tfCounter]),height=400,width=800)
            header <- sprintf("%s, %s\n(CF,IE,DR)=(%0.3f, %0.3f, %0.4f)",setName,tfName[tfCounter],snrFit$cf,snrFit$ie,snrFit$dr)
            snrPlotSdSig(snrFit,main=header)
            dev.off()
            # 1mer SNR
            plotHelper(sprintf("%ssnr.1mer.%s.png",plotDir,tfName[tfCounter]),height=400,width=800)
            header <- sprintf("%s, %s\n(CF,IE,DR)=(%0.3f, %0.3f, %0.4f)",setName,tfName[tfCounter],snrFit$cf,snrFit$ie,snrFit$dr)
            oneMerSNR(snrFit,doPlot=TRUE,main=header,ylim=snrLim)
            dev.off()
          }
        }
      }
    }
  }
}
