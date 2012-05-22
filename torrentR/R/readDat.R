readDat <- function(
  datFile,
  col=numeric(),
  row=numeric(),
  minCol=-1,
  maxCol=-1,
  minRow=-1,
  maxRow=-1,
  returnSignal=TRUE,
  returnWellMean=FALSE,
  returnWellSD=FALSE,
  returnWellLag=FALSE,
  uncompress=TRUE,
  doNormalize=FALSE,
  normStart=5,
  normEnd=20,
  XTCorrect=TRUE,
  chipType="",
  baselineMinTime=0,
  baselineMaxTime=0.7,
  loadMinTime=0,
  loadMaxTime=-1
) {
  
  nRegions = length(minCol)
  if(!all(c(length(minCol),length(maxCol),length(minRow),length(maxRow))==nRegions))
    stop("minCol,maxCol,minRow and maxRow must all be of the same length")

  callReadDat <- function(region) {
    val <- .Call("readDat",
      datFile,
      col,
      row,
      region[1],
      region[2],
      region[3],
      region[4],
      returnSignal,
      returnWellMean,
      returnWellSD,
      returnWellLag,
      uncompress,
      doNormalize,
      normStart,
      normEnd,
      XTCorrect,
      chipType,
      baselineMinTime,
      baselineMaxTime,
      loadMinTime,
      loadMaxTime,
      PACKAGE="torrentR"
    )
    nWell <- length(val$col)
    if((nWell > 0) && (!is.null(val$signal))) {
      val$signal <- matrix(val$signal,nrow=nWell)
    }
    if((nWell > 0) && (!is.null(val$wellMean))) {
      val$wellMean <- matrix(val$wellMean,nrow=nWell)
    }
    if((nWell > 0) && (!is.null(val$wellSD))) {
      val$wellSD <- matrix(val$wellSD,nrow=nWell)
    }
    if((nWell > 0) && (!is.null(val$wellLag))) {
      val$wellLag <- matrix(val$wellLag,nrow=nWell)
    }
    return(val)
  }

  regions <- split(c(minCol,maxCol,minRow,maxRow),rep(1:nRegions,4))
  allRegions <- lapply(regions,callReadDat)

  consolidateRegions <- function(i) {
    varsToNotCombine <- c(
      "datFile",
      "nCol",
      "nRow",
      "nFrame",
      "nFlow",
      "frameStart",
      "frameEnd"
    )
    varsToCombineAsVector <- c(
      "col",
      "row"
    )
    varsToCombineAsMatrix <- c(
      "signal",
      "wellMean",
      "wellSD",
      "wellLag"    
    )

    varName  <- names(allRegions[[1]])[i]
    if(is.element(varName,varsToNotCombine)) {
      return(allRegions[[1]][[i]])
    } else if(is.element(varName,varsToCombineAsVector)) {
      return(unlist(lapply(allRegions,function(z){z[[i]]}),use.names=FALSE))
    } else if(is.element(varName,varsToCombineAsMatrix)) {
      ncol <- ncol(allRegions[[1]][[i]])
      return(matrix(unlist(split(unlist(lapply(allRegions,function(z){t(z[[i]])}),use.names=FALSE),1:ncol),use.names=FALSE),ncol=ncol))
    }
  }

  if(nRegions==1) {
    res <- allRegions[[1]]
  } else {
    res <- lapply(1:length(allRegions[[1]]),consolidateRegions)
    names(res) <- names(allRegions[[1]])
  }

  return(res)
}
