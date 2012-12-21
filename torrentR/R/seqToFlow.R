seqToFlow <- function(sequence,flowOrder,nFlow=NA,finishAtSeqEnd=FALSE,flowOffset=0) {
  # Compute homopolymers
  sequence <- gsub("[^(ACGT)]","",sequence) # remove non-ACGT chars
  bases <- strsplit(sequence,"")[[1]]
  runEnds <- c(which(!bases[-length(bases)]==bases[-1]),length(bases))
  hpNuc <- bases[runEnds]
  hpLen <- diff(c(0,runEnds))
  hpN   <- length(hpNuc)

  # Compute flowOrder
  f <- strsplit(flowOrder,"")[[1]]
  if(is.na(nFlow))
    nFlow <- length(f)
  else
    f <- rep(f,ceiling(nFlow/length(f)))[1:nFlow]

  out <- rep(NA,nFlow)
  fIndex <- 0
  if(length(hpNuc)>0) {
    hpIndex <- 1
    for(fIndex in (1+flowOffset):nFlow) {
      if(hpNuc[hpIndex] == f[fIndex]) {
        out[fIndex] <- hpLen[hpIndex]
        hpIndex <- hpIndex + 1
        if(hpIndex > hpN)
          break;
      } else {
        out[fIndex] <- 0
      }
    }
  }
  if(fIndex < nFlow) {
    if(finishAtSeqEnd) {
      out <- out[-((fIndex+1):nFlow)]
    } else {
      out[(1+fIndex):nFlow] <- 0
    }
  }

  return(out)
}
