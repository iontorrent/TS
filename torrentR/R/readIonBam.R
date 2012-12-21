readIonBam <- function(
  bamFile,
  col=numeric(),
  row=numeric(),
  maxBases=250,
  nSample=0,
  randomSeed=1,
  readGroups=character(),
  wantMappingData=TRUE,
  maxCigarLength=100
) {

	if(!file.exists(bamFile))
		stop(sprintf("BAM file not found: %s",bamFile))
	if(length(col)!=length(row))
		stop("col and row arguments should be of the same length")
	if(maxBases < 0)
		stop("maxBases should be positive")
	if(randomSeed <= 0)
		stop("randomSeed should be positive")

	haveReadGroups <- length(readGroups) > 0
	val <- .Call("readIonBam",
	      bamFile, col, row, maxBases, nSample, randomSeed, readGroups, haveReadGroups, wantMappingData, maxCigarLength,
          PACKAGE="torrentR"
        )
	val$header <- readBamHeader(bamFile)
	return(val)
}
