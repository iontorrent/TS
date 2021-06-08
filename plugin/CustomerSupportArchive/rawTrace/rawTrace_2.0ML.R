
library(torrentR)
library(rjson)
set.seed(0)

jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
nucColor <- c(
  "A" = "green",
  "C" = "blue",
  "G" = "black",
  "T" = "red"
)
pngWidth <- 400
pngHeight <- 400
system(sprintf("mkdir -p %s",plotDir))

bfFiles  <- Sys.glob(sprintf("%s/beadfind_pre_*.dat",rawDir))
acqFiles <- Sys.glob(sprintf("%s/acq_*.dat",rawDir))
if(!is.na(acqLimit) && length(acqFiles > acqLimit)) {
  acqFiles <- acqFiles[1:acqLimit]
}
flowNum  <- 1+as.numeric(gsub(".dat","",gsub(".*_","",acqFiles)))
allFiles <- c(bfFiles,acqFiles)

makeDummyBf <- function(nCol,nRow) {
  bf <- list(
    nCol       = nCol,
    nRow       = nRow,
    col        = rep(0:(nCol-1),nRow),
    row        = rep(0:(nRow-1),rep(nCol,nRow)),
    maskEmpty  = rep(1,nCol*nRow),
    maskBead   = rep(0,nCol*nRow),
    maskPinned = rep(0,nCol*nRow),
    maskIgnore = rep(0,nCol*nRow)
  )
  return(bf)
}
if(exists("bfMaskFile")) {
  bf <- readBeadFindMask(bfMaskFile)
} else {
  temp <- readDat(allFiles[1],col=0,row=0,uncompress=FALSE)
  bf <- makeDummyBf(temp$nCol,temp$nRow)
}
bf$wellIndex <- bf$col + bf$row*bf$nCol 

regionW <- 25
regionH <- 25
if(chipType == "900") {
  # For bb we use 4 evenly-spaced regions along inlet-to-outlet axis
  nRegion = 3
  regionName = c("Inlet","Middle","Outlet")
  regionSep = 1/nRegion
  xVal = seq(regionSep/2,1-regionSep/2,length=nRegion)
  regionParam = lapply(1:length(xVal),function(z){
    list(
      name=regionName[z],
      x=xVal[z],
      y=0.5,
      w=regionW,
      h=regionH
    )
  })
} else {
  # 3-series chips
  lane = as.numeric(lane)
  if(lane > 0){  # multilane
    regionParam = list(
    list(name="Inlet",      x=0.25 * lane - 0.125, y=0.9, w=regionW, h=regionH),
    list(name="Middle",     x=0.25 * lane - 0.125, y=0.5, w=regionW, h=regionH),
    list(name="Outlet",     x=0.25 * lane - 0.125, y=0.1, w=regionW, h=regionH)
    #list(name="LowerRight", x=0.25 * lane - 0.125, y=0.1, w=regionW, h=regionH)
  )
  }else{
    regionParam = list(
    list(name="Inlet",      x=0.9, y=0.9, w=regionW, h=regionH),
    list(name="Middle",     x=0.5, y=0.5, w=regionW, h=regionH),
    list(name="Outlet",     x=0.1, y=0.1, w=regionW, h=regionH)
    #list(name="LowerRight", x=0.7, y=0.1, w=regionW, h=regionH)
  )
  }
  
}

nCol = bf$nCol
nRow = bf$nRow
makeRegion = function(r) {
  minCol = floor(r$x * nCol)
  minRow = floor(r$y * nRow)
  return(list(
    minCol=minCol,
    maxCol=minCol+r$w,
    minRow=minRow,
    maxRow=minRow+r$h,
    width=r$w,
    height=r$h,
    name=r$name
  ))
}
regions = lapply(regionParam,makeRegion)

# Add a region for the best alignment, if we have a best alignment
haveBestBead <- FALSE
# if(!is.na(samParsedFile)) {
#   sam <- readSamParsed(samParsedFile,fields=c("name",paste("q",as.character(c(7,10,17,20,47)),"Len",sep="")))
#   if(length(sam[[1]]) > 0) {
#     haveBestBead <- TRUE
#     iWell <- which.max(sam$q47Len)[1]
#     bestCol <- sam$col[iWell]
#     bestRow <- sam$row[iWell]
#     bestLen <- sam$q47Len[iWell]
#     wellIndex <- bestCol + bestRow*bf$nCol
#     bestRegionW <- 50
#     bestRegionH <- 50
#     bestRegionMinCol <- 50 *floor(bestCol / 50)
#     bestRegionMinRow <- 50 *floor(bestRow / 50)
#     newRegion <- list(
#       minCol = bestRegionMinCol,
#       minRow = bestRegionMinRow,
#       maxCol = bestRegionMinCol + bestRegionW,
#       maxRow = bestRegionMinRow + bestRegionH,
#       width  = bestRegionW,
#       height = bestRegionH,
#       name   = "bestBead"
#     )
#     regions <- c(list(newRegion),regions)
#   }
# }

names(regions) <- unlist(lapply(regions,function(z){z$name}))

processDat <- function(datFile,regions) {
  minCol <- unlist(lapply(regions,function(z){z$minCol}))
  maxCol <- unlist(lapply(regions,function(z){z$maxCol}))
  minRow <- unlist(lapply(regions,function(z){z$minRow}))
  maxRow <- unlist(lapply(regions,function(z){z$maxRow}))
  d <- readDat(datFile,minCol=minCol,maxCol=maxCol,minRow=minRow,maxRow=maxRow,uncompress=FALSE)
  regionSize <- (maxCol-minCol) * (maxRow-minRow)
  d$regionIndex <- apply(cbind(cumsum(regionSize),c(1,1+cumsum(regionSize))[1:length(regions)]),1,function(z){seq(z[2],z[1])})
  if(mode(d$regionIndex)=="numeric") {
    # apply can return a list or matrix depending on whether the regions are
    # the same size or not.  So here we ensure it is always a list
    d$regionIndex <- split(d$regionIndex,rep(1:ncol(d$regionIndex),rep(nrow(d$regionIndex),ncol(d$regionIndex))))
    names(d$regionIndex) <- names(regions)
  }
  d$wellIndex <- d$col + d$row*d$nCol
  if(haveBestBead) {
    d$indexGood  <- which(d$col==bestCol & d$row==bestRow)
  }
  d$mask <- list(
    empty   = is.element(d$wellIndex,(bf$wellIndex)[bf$maskEmpty ==1]),
    bead    = is.element(d$wellIndex,(bf$wellIndex)[bf$maskBead  ==1]),
    pinned  = is.element(d$wellIndex,(bf$wellIndex)[bf$maskPinned==1]),
    ignore  = is.element(d$col + d$row*d$nCol,(bf$wellIndex)[bf$maskIgnore==1])
  )

  d$bkg <- lapply(as.list(names(regions)),function(regionName) {
    regionMask <- rep(FALSE,length(d$mask$empty))
    regionMask[d$regionIndex[[regionName]]] <- TRUE
    bkgMask <- d$mask$empty & regionMask
    if(any(bkgMask)) {
      return(apply(d$signal[bkgMask,],2,mean))
    } else {
      return(rep(0,ncol(d$signal)))
    }
  })
  names(d$bkg) <- names(regions)

  return(d)
}

# Read all the data
allDat <- lapply(as.list(allFiles),processDat,regions=regions)
max_start = 0
max_end = 0
frameStart  <- allDat[[1]]$frameStart
frameEnd    <- allDat[[1]]$frameEnd
for(i in 1:length(allDat)){
  if(length(allDat[[i]]$frameStart) > max_start){
    frameStart = allDat[[i]]$frameStart
    max_start = length(allDat[[i]]$frameStart)
  }
  if(length(allDat[[i]]$frameEnd) > max_end){
    frameEnd = allDat[[i]]$frameEnd
    max_end = length(allDat[[i]]$frameEnd)
  }
}
regionIndex <- allDat[[1]]$regionIndex

if(haveBestBead) {
  bestSig <- as.list(1:length(allFiles))
}
beadSig <- as.list(1:length(allFiles))
for(i in 1:length(allFiles)) {
  datFile <- allFiles[i]
  flowName <- gsub(".*/","",datFile)
  flowName <- gsub(".dat","",flowName)
  x <- allDat[[i]]
  beadSig[[i]] <- lapply(names(regions),function(regionName){
    regionMask <- rep(FALSE,length(x$mask$empty))
    regionMask[x$regionIndex[[regionName]]] <- TRUE
    apply(x$signal[x$mask$bead & regionMask,],2,mean)-x$bkg[[regionName]]
  })
  names(beadSig[[i]]) <- names(regions)

  if(haveBestBead) {
    bestRegionBkg <- x$bkg$bestBead
    bestSig[[i]] <- x$signal[x$indexGood,]-bestRegionBkg
    yOneBead <- bestSig[[i]]
    yAllBead <- beadSig$bestBead

    ylim <- range(c(yOneBead,yAllBead))
    png(plotFile <- sprintf("%s/%s.png",plotDir,flowName),height=pngHeight,width=pngWidth)
    par(mfrow=c(2,1),mai=c(0.35,0.75,0.05,0.05),omd=c(0.03,0.98,0.05,0.9))
    plot(frameStart,yAllBead,ylim=ylim,col="blue",type="l",xaxt="n",ylab="Bkg-subtracted Signal",las=2)
    points(frameStart,yOneBead,bg="red",pch=21)
    lines(frameStart,yOneBead,col="red",lwd=3)
    plot(frameStart,bestRegionBkg,xlab="Time (s)",ylab="Bkg Signal",type="l",lwd=3,las=2)
    legend("bottomright",inset=0.01,c(sprintf("(c%d,r%d)",bestCol,bestRow),"All Beads"),fill=c("red","blue"),bty="n")
    mtext(sprintf("%s %s",analysisName,flowName),side=3,outer=TRUE,line=1)
    dev.off()
    #system(sprintf("eog %s",plotFile))
  }
}

floworderExpanded <- rep(unlist(strsplit(floworder,"")),ceiling(max(flowNum)/nchar(floworder)))
flowBase <- c(rep(NA,length(bfFiles)),floworderExpanded[flowNum])
flowBaseNum <- c(rep(NA,length(bfFiles)),flowNum)
signalBeadFind <- grep("beadfind_pre_0004.dat",bfFiles)
if(length(signalBeadFind) > 0) {
  flowBase[signalBeadFind] <- "G"
  flowBaseNum[signalBeadFind] <- 0
}

# Individual nuc steps
ylim <- range(c(lapply(allDat[!is.na(flowBase)],function(z){unlist(z$bkg)})))
for(iRegion in 1:length(regions)) {
  regionName <- names(regions)[iRegion]
  bkg <- lapply(allDat,function(z){z$bkg[[regionName]]})
  bkgSplit <- split(bkg,flowBase)
  png(plotFile <- sprintf("%s/nucSteps.%s.png",plotDir,regionName),height=pngHeight,width=2*pngWidth)
  par(mfrow=c(1,4),mai=c(0.2,0.2,0.15,0.05),omd=c(0.03,0.98,0.05,0.9))
  for(nuc in c("A","C","G","T")) {
    thisBkg <- bkgSplit[[nuc]]
    if(is.element(nuc,c("A"))) {
      ylab="Bkg Signal"
      yaxt="s"
    } else {
      ylab=""
      yaxt="n"
    }
    if(is.element(nuc,c("A","C","G","T"))) {
      xlab="Time (s)"
      xaxt="s"
    } else {
      xlab=""
      xaxt="n"
    }
    plot(range(frameStart),main=nuc,ylim,type="n",xlab=xlab,xaxt=xaxt,ylab=ylab,yaxt=yaxt)
    abline(v=frameStart,col="lightgrey")
    thisColor <- jet.colors(length(thisBkg))
    for(i in 1:length(thisBkg)) {
      lines(frameStart,thisBkg[[i]],col=thisColor[i])
    }
  }
  mtext(sprintf("Nuc Steps for %s\n%dx%d region, lower-left corner at (c%d,r%d)\n%d empties and %d beads",
    analysisName,
      regions[[iRegion]]$width,regions[[iRegion]]$height,
    regions[[iRegion]]$minCol,regions[[iRegion]]$minRow,
    sum(allDat[[1]]$mask$empty[regionIndex[[regionName]]]),
    sum(allDat[[1]]$mask$bead[regionIndex[[regionName]]])
  ),outer=TRUE,side=3,line=0)
  dev.off()
  #system(sprintf("eog %s",plotFile))
}

# nuc step sizes
ylim <- range(c(lapply(allDat[!is.na(flowBase)],function(z){c(-1,1) %*% sapply(z$bkg,range)})))
for(iRegion in 1:length(regions)) {
  regionName <- names(regions)[iRegion]
  bkg <- lapply(allDat,function(z){z$bkg[[regionName]]})
  stepSize <- c(c(-1,1) %*% sapply(bkg,range))
  nucSplit <- split(1:length(stepSize),flowBase)
  png(plotFile <- sprintf("%s/nucStepSize.%s.png",plotDir,regionName),height=pngHeight,width=pngWidth)
  plot(range(flowBaseNum,na.rm=TRUE),ylim,xlab="Flow",ylab="Step Size",type="n")
  for(nuc in c("A","C","G","T")) {
    flowsSinceLast <- c(0,diff(flowBaseNum[nucSplit[[nuc]]]))
    lines(flowBaseNum[nucSplit[[nuc]]],stepSize[nucSplit[[nuc]]],col=nucColor[nuc],type="b",pch=as.character(flowsSinceLast))
  }
  title(sprintf("Nuc Step Sizes, %s\nvalue = flows since same nuc was last flowed",regionName))
  legend("topright",inset=0.01,c("A","C","G","T"),fill=nucColor,bty="n")
  dev.off()
  #system(sprintf("eog %s",plotFile))
}

# Key traces
if(nchar(libKey) > 1 && nchar(floworder) > 1) {
  idealKey <- seqToFlow(libKey,paste(rep(floworder,10),collapse=""),finishAtSeqEnd=TRUE)
  keyNucs <- unlist(strsplit(paste(rep(floworder,3),collapse=""),""))[1:length(idealKey)]
  posMerNucs  <- keyNucs[which(idealKey>0)]
  zeroMerNucs <- keyNucs[which(idealKey==1)]
  allNucs <- c(unique(posMerNucs),unique(zeroMerNucs))
  nBF <- length(bfFiles)

  # Zeromer-subtracted traces
  if(any(duplicated(allNucs))) {
    allNucs <- sort(allNucs[duplicated(allNucs)])
    zeroMerBead <- as.list(allNucs)
    names(zeroMerBead) <- allNucs
    zeroMerBest <- zeroMerBead
    for(iRegion in 1:length(regions)) {
      regionName <- names(regions)[iRegion]
      for(iNuc in 1:length(allNucs)) {
        thisNuc <- allNucs[iNuc]
        nucFlows <- which(idealKey==0 & keyNucs==thisNuc)
        zeroMerBead[[iNuc]] <- apply(matrix(unlist((lapply(beadSig,function(z){z[[regionName]]}))[ nBF+nucFlows]),ncol=length(nucFlows)),1,mean)
        if(regionName=="bestBead") {
          zeroMerBest[[iNuc]] <- apply(matrix(unlist(bestSig[nBF+nucFlows]),ncol=length(nucFlows)),1,mean)
        }
      }
      zeroSubBead <- zeroSubBest <- list()
      zeroSubNuc <- character()
      for(iNuc in 1:length(allNucs)) {
        thisNuc <- allNucs[iNuc]
        nucFlows <- which(idealKey>0 & keyNucs==thisNuc)
        for(nucFlow in nucFlows) {
          zeroSubBead <- c(zeroSubBead, list((lapply(beadSig,function(z){z[[regionName]]}))[[ nBF+nucFlow]]-zeroMerBead[[ thisNuc]]))
          if(regionName=="bestBead") {
            zeroSubBest <- c(zeroSubBest, list(bestSig[[nBF+nucFlow]]-zeroMerBest[[thisNuc]]))
          }
          zeroSubNuc <- c(zeroSubNuc,thisNuc)
        }
      }
      if(any(!is.nan(unlist(zeroSubBead)))) {
        outFileBase <- sprintf("%s/keyZeroSubBead.%s",plotDir,regionName)
        plotFile <- sprintf("%s.png",outFileBase)
        jsonZeroSubBeadFile <- sprintf("%s.json",outFileBase)
        png(plotFile,height=pngHeight,width=pngWidth)
        ylim <- range(unlist(zeroSubBead))
        plot(range(frameStart),ylim,type="n",xlab="Time (s)",ylab="Zeromer-Subtracted Signal",las=2)
        for(i in 1:length(zeroSubBead)) {
          lines(frameStart,zeroSubBead[[i]],col=nucColor[zeroSubNuc[i]],lwd=2)
        }
        title(sprintf("Zeromer-subtracted key traces\n%d bead and %d empty wells\n%dx%d region, lower-left at (c%d,r%d)",
          sum(allDat[[1]]$mask$bead[regionIndex[[regionName]]]),sum(allDat[[1]]$mask$empty[regionIndex[[regionName]]]),
          regions[[iRegion]]$width,regions[[iRegion]]$height,
          regions[[iRegion]]$minCol,regions[[iRegion]]$minRow)
        )
        legend("topright",inset=0.01,c("A","C","G","T"),fill=nucColor,bty="n")
        dev.off()
        write(toJSON(
          list(
            "frameStart"    = c(frameStart),
            "frameEnd"      = c(frameEnd),
            "nuc"           = zeroSubNuc,
            "incorporation" = zeroSubBead
          )
        ),file=jsonZeroSubBeadFile)
        #system(sprintf("eog %s",plotFile))
      }
  
      if(regionName=="bestBead") {
        if(any(!is.nan(unlist(zeroSubBest)))) {
          png(plotFile <- sprintf("%s/keyZeroSubBest.png",plotDir),height=pngHeight,width=pngWidth)
          ylim <- range(unlist(zeroSubBest))
          plot(range(frameStart),ylim,type="n",xlab="Time (s)",ylab="Zeromer-Subtracted Signal",las=2)
          for(i in 1:length(zeroSubBest)) {
            lines(frameStart,zeroSubBest[[i]],col=nucColor[zeroSubNuc[i]],lwd=2)
          }
          title(sprintf("Zeromer-subtracted key traces\nBead (c%d,r%d), Q47Len=%d",bestCol,bestRow,bestLen))
          legend("topright",inset=0.01,c("A","C","G","T"),fill=nucColor,bty="n")
          dev.off()
          #system(sprintf("eog %s",plotFile))
        }
      }
    }
  }

  keyFlows <- length(bfFiles)+(1:length(idealKey))
  if(!all(is.nan(unlist(beadSig[keyFlows])))) {
    temp <- unlist(beadSig[keyFlows])
    ylim <- range(temp[!is.nan(temp)])
    for(iRegion in 1:length(regions)) {
      regionName <- names(regions)[iRegion]
      if(regionName=="bestBead") {
        png(plotFile <- sprintf("%s/bestKey.png",plotDir,regionName),height=pngHeight,width=pngWidth)
        bestLim <- range(unlist(bestSig[keyFlows]))
        plot(range(frameStart),bestLim,type="n",xlab="Time (s)",ylab="Bkg-Subtracted Signal",las=2)
        title(sprintf("Bead (c%d,r%d), Q47Len=%d",bestCol,bestRow,bestLen))
        for(iFlow in keyFlows) {
          lines(frameStart,bestSig[[iFlow]],col=nucColor[flowBase[iFlow]],lwd=2)
        }
        dev.off()
        #system(sprintf("eog %s",plotFile))
      }

      png(plotFile <- sprintf("%s/keyFlows.%s.png",plotDir,regionName),height=pngHeight,width=pngWidth)
      plot(range(frameStart),ylim,type="n",xlab="Time (s)",ylab="Bkg-Subtracted Signal",las=2)
      title(sprintf("Average of all %d beads\n%dx%d region, lower-left at (c%d,r%d)",
        sum(allDat[[1]]$mask$bead[regionIndex[[regionName]]]),
        regions[[iRegion]]$width,regions[[iRegion]]$height,
        regions[[iRegion]]$minCol,regions[[iRegion]]$minRow)
      )
      for(iFlow in keyFlows) {
        lines(frameStart,beadSig[[iFlow]][[regionName]],col=nucColor[flowBase[iFlow]],lwd=2)
      }
      dev.off()
      #system(sprintf("eog %s",plotFile))
    }
  }
}

