treePhaser <- function(
  signal,
  flowOrder,
  cf,
  ie,
  dr,
  keySeq="",
  basecaller=c("treephaser-swan", "dp-treephaser", "treephaser-adaptive", "treephaser-solve"),
  diagonalStates=0,
  RecalModelFile="",
  RecalModelThreshold=4,
  xval=NA,
  yval=NA
) {

  basecaller <- match.arg(basecaller)

  if(!is.matrix(signal))
    signal <- matrix(signal,nrow=1)

  if(nchar(flowOrder) < ncol(signal))
    flowOrder <- substring(paste(rep(flowOrder,ceiling(ncol(signal)/nchar(flowOrder))),collapse=""),1,ncol(signal))

  if(keySeq=="") {
    keyFlow <- numeric()
  } else {
    keyFlow <- seqToFlow(keySeq,flowOrder,finishAtSeqEnd=TRUE)
  }
  
  # Number of phase parameters must equal to the number of reads or 1 value for all
  if ((length(cf) != 1) && (length(cf) != nrow(signal)))
    stop("Error in treephaser: Lenght of <cf> must be 1 or equal to nrow(signal)")
  if (length(ie) != length(cf))
    stop("Error in treephaser: Length of <ie> must be equal to length of cf.")
  if (length(dr) != length(cf))
    stop("Error in treephaser: Length of <dr> must be equal to length of cf.")
    
  if (RecalModelFile!="") {
    if (any(is.na(xval)) | (length(xval) != nrow(signal)))
      stop("Error in treephaser: If recalibration file is provided <xval> and <yval> need to be provided for every read.")
    if (any(is.na(yval)) | (length(yval) != nrow(signal)))
      stop("Error in treephaser: If recalibration file is provided <xval> and <yval> need to be provided for every read.")
  }
    
  val <- .Call("treePhaser", signal, keyFlow, flowOrder, cf, ie, dr, basecaller, diagonalStates, RecalModelFile, RecalModelThreshold, xval, yval, PACKAGE="torrentR")

  return(val)
}
