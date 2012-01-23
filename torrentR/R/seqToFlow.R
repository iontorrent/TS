seqToFlow <- function(sequence,flowOrder,nFlow=NA,finishAtSeqEnd=FALSE) {
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
  hpIndex <- 1
  for(fIndex in 1:nFlow) {
    if(hpNuc[hpIndex] == f[fIndex]) {
      out[fIndex] <- hpLen[hpIndex]
      hpIndex <- hpIndex + 1
      if(hpIndex > hpN)
        break;
    } else {
      out[fIndex] <- 0
    }
  }
  if(finishAtSeqEnd) {
    out <- out[!is.na(out)]
  } else {
    out[is.na(out)] <- 0
  }

  return(out)
}
