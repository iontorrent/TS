readTfInfo <- function(
  dataDir,
  tfKeySeq,
  minTfNumber=1000
) {

  tfInfo <- list()

  # Load up the library of valid TF sequences
  tfSet <- readTfConf(dataDir)
  if(is.null(tfSet)) {
    cat(sprintf("Unable to read TF config for %s\n",dataDir))
  } else {
    # Load up the data on the TF sequences available
    tfStats <- readTfStats(dataDir)
    if(is.null(tfStats)) {
      cat(sprintf("Unable to read TF tracking data for %s\n",dataDir))
    } else if(length(tfStats[[1]])==0) {
      cat(sprintf("No TFs found for %s\n",dataDir))
    } else {
      tfFound <- names(which(table(tfStats$tfSeq) > minTfNumber))
      nTfFound <- length(tfFound)
      if(nTfFound==0) {
        cat(sprintf("Did not find any TFs with at least %d instances for %s\n",minTfNumber,dataDir))
      } else {
        if(nTfFound > 1)
          cat(sprintf("Found %d TFs with at least %d instances for %s\n",nTfFound,minTfNumber,dataDir))
        tfInfo <- as.list(1:nTfFound)
        names(tfInfo) <- tfFound
        for(tfCounter in 1:nTfFound) {
          seqIndex <- which(tfSet$tf==tfFound[tfCounter] & tfSet$keySeq==tfKeySeq)
          if(length(seqIndex)==0) {
            warning(sprintf("Failed to find tfSeq %s with key %s in library, skipping...\n",tfFound,tfKeySeq))
          } else {
            if(length(seqIndex)>1) {
              warning(sprintf("Found %d entries matching tfSeq %s with key %s in library, using only the first...\n",length(seqIndex),tfFound,tfKeySeq))
              seqIndex <- seqIndex[1]
            }
            tfMask <- (tfStats$tfSeq==tfSet$tf[seqIndex])
            tfInfo[[tfCounter]] <- list(
              n = sum(tfMask),
              seq = paste(tfKeySeq,tfSet$tfSeq[seqIndex],sep=""),
              row = tfStats$row[tfMask],
              col = tfStats$col[tfMask]
            )
          }
        }
      }
    }
  }

  return(tfInfo)
}
