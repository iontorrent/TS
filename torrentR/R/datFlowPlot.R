datFlowPlot <- function(
  analysisDir,
  tfKeySeq,
  datDir=NA,
  plotDir=NA,
  setName=NA,
  minWellNumber=5000,
  nColBin=50,
  nRowBin=50,
  binTarget=50,
  maxFlows=5,
  plotType=c("png","bitmap","pdf","none"),
  ret=FALSE
) {

  plotType <- match.arg(plotType)

  # Set random seed for reproducible sampling
  set.seed(0)

  # If setName wasn't specified use the base name of the data directory
  if(is.na(setName))
    setName <- basename(analysisDir)
          
  # Add trailing file separators if not already present
  if(.Platform$file.sep != rev(unlist(strsplit(analysisDir,"")))[1])
    analysisDir <- sprintf("%s%s",analysisDir,.Platform$file.sep)
  if(plotType != "none") {
    if(.Platform$file.sep != rev(unlist(strsplit(plotDir,"")))[1])
      plotDir <- sprintf("%s%s",plotDir,.Platform$file.sep)
  }

  bfMaskFile <- paste(analysisDir,"bfmask.bin",sep="")
  bf <- readBeadFindMask(bfMaskFile)
  pctLoaded  <- mean(bf$maskBead)

  # Find the location for the dat files, if it wasn't specified
  if(is.na(datDir))
    datDir <- findDatDir(analysisDir)
  if(is.null(datDir)) {
    warning(sprintf("Unable to locate dat directory for %s\n",analysisDir))
  } else {
    datFile <- readDatList(datDir)$datFiles
    if(length(datFile)==0) {
      warning(sprintf("Unable to locate any dat images in directory %s\n",datDir))
    } else {
      colEmpty <- bf$col[bf$maskEmpty==1]
      rowEmpty <- bf$row[bf$maskEmpty==1]
      
      warning.readTfConfig    <- FALSE
      warning.readTfTracking  <- FALSE
      warning.noTfFound       <- FALSE
      warning.insufficientTf  <- FALSE
      warning.insufficientLib <- FALSE
      for(datIndex in 1:min(length(datFile),maxFlows)) {
        thisDat <- sprintf("%s%s%s",datDir,.Platform$file.sep,datFile[datIndex])
        dat <- readDat(thisDat)

        # Plot the background for every frame and the summary traces, interpolate to all positions
        datEmpty <- dat$signal[bf$maskEmpty==1,]
        baseName <- sprintf("%s%s.baselined.empty",plotDir,datFile[datIndex])
        header <- sprintf("%s\n%s; baselined empties",analysisDir,datFile[datIndex])
        cat(sprintf("  Writing baselined flowplots for empty wells & computing background\n"))
        background <- flowFrameTracePlot(
          colEmpty,
          rowEmpty,
          datEmpty,
          baseName,
          1:dat$nFrame,
          header,
          bf$nCol,
          bf$nRow,
          nColBin,
          nRowBin,
          plotType=plotType,
          doInterpolate=TRUE
        )

        # Now make the background-subtracted signal and plot it
        bkgSub <- dat$signal - background

        # Plot the background-subtracted empties for every frame
        cat(sprintf("  Writing background-subtracted flowplots for empty wells\n"))
        baseName <- sprintf("%s%s.bkgSub.empty",plotDir,datFile[datIndex])
        header <- sprintf("%s\nbkgSub empties; %s",analysisDir,datFile[datIndex])
        flowFrameTracePlot(
          colEmpty,
          rowEmpty,
          bkgSub[bf$maskEmpty==1,],
          baseName,
          1:dat$nFrame,
          header,
          bf$nCol,
          bf$nRow,
          nColBin,
          nRowBin,
          plotType=plotType,
          doInterpolate=FALSE
        )

        # Plot background-subtracted library beads
        baseName <- sprintf("%s%s.bkgSub.lib",plotDir,datFile[datIndex])
        header <- sprintf("%s\nbkgSub lib; %s",analysisDir,datFile[datIndex])
        libMask <- bf$maskLive & bf$maskLib & !(bf$maskWashout | bf$maskIgnore)
        if(sum(libMask) < minWellNumber) {
          warning.insufficientLib <- TRUE
        } else {
          flowNumber <- as.numeric(gsub(".dat","",gsub("acq_","",datFile[datIndex])))+1
          wellSignal <- readWells(sprintf("%s1.wells",analysisDir),col=bf$col[libMask],row=bf$row[libMask],flow=flowNumber)$signal
          nBins <- floor(sqrt(sum(libMask)/binTarget))
          cat(sprintf("  Writing background-subtracted flowplots for library wells\n"))
          flowFrameTracePlot(
            bf$col[libMask],
            bf$row[libMask],
            bkgSub[libMask,],
            baseName,
            1:dat$nFrame,
            header,
            bf$nCol,
            bf$nRow,
            nBins,
            nBins,
            plotType=plotType,
            minBin=10,
            doInterpolate=FALSE,
            wellSignal=wellSignal
          )
        }

        # Look for TFs and plot any that are found
        tfSet <- readTfConf(analysisDir)
        if(is.null(tfSet)) {
          warning.readTfConfig <- TRUE
        } else {
          tfStats <- readTfStats(analysisDir)
          if(is.null(tfStats)) {
            warning.readTfTracking <- TRUE
          } else if(length(tfStats[[1]])==0) {
            warning.noTfFound <- TRUE
          } else {
            tfName <- sort(names(which(table(tfStats$tfSeq) > minWellNumber)))
            nTfTypes <- length(tfName)
            if(nTfTypes==0) {
              warning.insufficientTf <- TRUE
            } else {
              for(tfCounter in 1:nTfTypes) {
                tfMask <- tfStats$tfSeq==tfName[tfCounter]
                nTF <- sum(tfMask)
                flowNumber <- as.numeric(gsub(".dat","",gsub("acq_","",datFile[datIndex])))+1
                x <- readWells(sprintf("%s1.wells",analysisDir),col=tfStats$col[tfMask],row=tfStats$row[tfMask],flow=flowNumber)
                if(sum(temp <- tfSet$tf==tfName[tfCounter] & tfSet$keySeq==tfKeySeq) == 1) {
                  trueSeq <- paste(tfSet$keySeq[temp],tfSet$tfSeq[temp],sep="")
                  trueFlow <- seqToFlow(trueSeq,x$flowOrder)
                } else {
                  trueSeq <- NULL
                  trueFlow <- NULL
                }

                # Plot the background for every flow, interpolate to all flow
                baseName <- sprintf("%s%s.bkgSub.tf_%s",plotDir,datFile[datIndex],tfName[tfCounter])
                header <- sprintf("%s\nbkgSub tf %s; %s",analysisDir,tfName[tfCounter],datFile[datIndex])
                thisTfIndex <- tfStats$row[tfMask]*bf$nCol + tfStats$col[tfMask] + 1
                nBins <- floor(sqrt(nTF/binTarget))
                cat(sprintf("  Writing background-subtracted flowplots for %s\n",tfName[tfCounter]))
                flowFrameTracePlot(
                  tfStats$col[tfMask],
                  tfStats$row[tfMask],
                  bkgSub[thisTfIndex,],
                  baseName,
                  1:dat$nFrame,
                  header,
                  bf$nCol,
                  bf$nRow,
                  nBins,
                  nBins,
                  plotType=plotType,
                  minBin=10,
                  doInterpolate=FALSE,
                  wellSignal=x$signal
                )
              }
            }
          }
        }
      }
      if(warning.readTfConfig)
        warning(sprintf("Unable to read TF config for %s\n",analysisDir))
      if(warning.readTfTracking)
        warning(sprintf("Unable to read TF tracking data for %s\n",analysisDir))
      if(warning.noTfFound)
        warning(sprintf("No TFs found for %s\n",analysisDir))
      if(warning.insufficientTf)
        warning(sprintf("Insufficient TFs for %s (need %d, most frequent was %d.  Skipping...\n",analysisDir,minWellNumber,max(table(tfStats$tfSeq))))
      if(warning.insufficientLib)
        warning(sprintf("Insufficient lib beads for %s (need %d, found %d.  Skipping...\n",analysisDir,minWellNumber,sum(libMask)))
    }
  }

  if(ret) {
    return(list(
      dat=dat,
      background=background,
      bf=bf
    ))
  }
}
