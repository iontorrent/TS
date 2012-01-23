findLibReads <- function(bf,wellFile,keyFlowLen,libKeySeq,nSample=10000,keySnrThreshold=3) {
  # Use the bfmask info if available.  For older runs it won't be, in which case it has to be computed
  # Commenting out the use of maskKeypass because it looks like it might be broken at the moment
  #if(is.null(bf$maskKeypass)) {
    x <- readWells(wellFile,flow=1:keyFlowLen)
    possibleIndex <- which(x$mask$lib & !(x$mask$ignore | x$mask$washout))
    cat(sprintf("Looking at key signals for %d possible lib beads\n",length(possibleIndex)))
    keyStat <- keyStats(x$signal[possibleIndex,],libKeySeq,x$flowOrder)
    goodSignal <- (keyStat$key_snr > keySnrThreshold)
    goodSignal[is.na(goodSignal)] <- FALSE
    index <- possibleIndex[goodSignal]
    keyPassPct <- mean(goodSignal)
    cat(sprintf("  %0.1f%% have a key SNR greater than %0.1f\n",100*keyPassPct,keySnrThreshold))
  #} else {
  #  index <- which(bf$maskKeypass==1)
  #  keyPassPct <- sum(bf$maskKeypass) / sum(bf$maskLib & !(bf$maskIgnore | bf$maskWashout))
  #}
  return(list(index=index,keyPassPct=keyPassPct))
}
