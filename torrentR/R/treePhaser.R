treePhaser <- function(
  signal,
  flowOrder,
  cf,
  ie,
  dr,
  keySeq="",
  basecaller=c("treephaser-swan", "dp-treephaser", "treephaser-adaptive"),
  terminatorChemistryRun=0
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
  val <- .Call("treePhaser", signal, keyFlow, flowOrder, cf, ie, dr, basecaller, terminatorChemistryRun, PACKAGE="torrentR")

  return(val)
}
