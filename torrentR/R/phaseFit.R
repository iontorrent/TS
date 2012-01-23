phaseFit <- function(
  trueSeq,
  sig,
  flowOrder,
  cf                 = 0.0100,
  ie                 = 0.0050,
  dr                 = 0.0015,
  hpScale            = 1,
  conc               = diag(4),
  maxAdvances        = 2,
  maxIter            = 30,
  droopType          = c("ONLY_WHEN_INCORPORATING","EVERY_FLOW"),
  fitType            = c("CfIeDr","CfIeDrHpScale","HpScale","CfIeDrHpScale4","HpScale4","CfIe","NucContam","CfIe4","NucContamIe"),
  resType            = c("SQUARED","ABSOLUTE","GEMAN_MCCLURE"),
  resSummary         = c("MEAN","MEDIAN","MEAN_OF_MEDIAN","MEDIAN_OF_MEAN"),
  ignoreHPs          = FALSE,
  flowWeight         = NULL,
  maxErr             = 1,
  extraTaps          = 0
) {

  droopType  <- match.arg(droopType)
  fitType    <- match.arg(fitType)
  resType    <- match.arg(resType)
  resSummary <- match.arg(resSummary)

  if((length(hpScale) != 1) & (length(hpScale) != 4))
    stop("hpScale must be of length 1 or 4\n");

  # trueSeq can be either a vector of strings or a numeric matrix - handle whichever was supplied
  if(mode(trueSeq)=="character") {
    seqString <- trueSeq
    seqFlow   <- matrix(0,0,0)
    nSeq      <- length(seqString)
    nFlow     <- NA
  } else if(mode(trueSeq)=="numeric") {
    if(!is.matrix(trueSeq))
      trueSeq <- matrix(trueSeq,nrow=1)
    seqString <- ""
    seqFlow   <- trueSeq
    nSeq      <- nrow(seqFlow)
    nFlow     <- ncol(seqFlow)
  } else {
    stop("first argument must be either a vector of dna strings or a matrix of true flow values")
  }

  if(!is.matrix(sig))
    sig <- matrix(sig,nrow=1)
  if(nrow(sig) != nSeq)
    stop("signal matrix must have one row for every sequence")
  if(is.na(nFlow)) {
    nFlow <- ncol(sig)
  } else if(ncol(sig) != nFlow) {
    stop("signal matrix must have one column for every flow")
  }

  if(is.null(flowWeight)) {
    flowWeight <- rep(1,nFlow)
  } else if(length(flowWeight) != nFlow) {
    stop("flowWeight should be the same length as nFlow");
  }

  val <- .Call("phaseFit", seqString, seqFlow, sig, flowOrder, conc, cf, ie, dr, hpScale, nFlow, maxAdvances, droopType, maxIter, fitType, ignoreHPs, flowWeight, resType, resSummary, maxErr, extraTaps, PACKAGE="torrentR")

  if(any(names(val)=="conc"))
    dimnames(val$conc) <- c(rep(list(c("A","C","G","T")),2))

  return(val)
}
