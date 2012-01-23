findBestCafie <- function(
  signal,
  flowOrder,
  keySeq,
  trueSeq="",
  cf=-1,
  ie=-1,
  dr=-1,
  analysisMode=c("knownSeq","knownCAFIE"),
  combine=TRUE,
  doKeyNorm=TRUE,
  doScale=FALSE,
  hpSignal=0:12,
  sigMult=1
) {

	analysisMode <- match.arg(analysisMode)

	if(analysisMode == "knownSeq") {
	  if(length(trueSeq)==0)
	    stop("must supply value for trueSeq if analysisMode is knownSeq")
	} else {
	  if(cf < 0)
	    stop("must supply value for cf if analysisMode is knownCAFIE")
	  if(ie < 0)
	    stop("must supply value for ie if analysisMode is knownCAFIE")
	  if(dr < 0)
	    stop("must supply value for dr if analysisMode is knownCAFIE")
	}

	if(doScale)
	  doScale <- 1
        else
	  doScale <- 0

	if(doKeyNorm)
	  doKeyNorm <- 1
        else
	  doKeyNorm <- 0

	# If combining signals, take a median
	if(combine & !is.null(dim(signal)))
	    signal <- apply(signal,2,median)

	# If signal is not a matrix, make it so
	if(is.null(dim(signal)))
	  signal <- matrix(signal,1,length(signal))
	nFlow <- ncol(signal)
  
	# Cycle flowOrder if necessary
        if(nchar(flowOrder) < nFlow)
          flowOrder <- substring(paste(rep(flowOrder,ceiling(nFlow/nchar(flowOrder))),collapse=""),1,nFlow)

	# Compute the keyFlow by comparing keySeq with flowOrder
	keyFlow <- seqToFlow(keySeq,flowOrder,finishAtSeqEnd=TRUE)
        nKeyFlow <- length(keyFlow)-1

	val <- .Call("findBestCafie",
          signal,
          flowOrder,
          keyFlow,
          trueSeq,
          cf,
          ie,
          dr,
	  analysisMode,
          nKeyFlow,
          doKeyNorm,
          doScale,
	  hpSignal,
	  sigMult,
          PACKAGE="torrentR")
	val
}
