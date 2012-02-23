readSFF <- function(
  sffFile,
  col=numeric(),
  row=numeric(),
  maxBases=250,
  nSample=0,
  randomSeed=1
) {

	if(!file.exists(sffFile))
		stop(sprintf("SFF file not found: %s",sffFile))
	if(length(col)!=length(row))
		stop("col and row arguments should be of the same length")
	if(maxBases < 0)
		stop("maxBases should be positive")
	if(randomSeed <= 0)
		stop("randomSeed should be positive")

	val <- .Call("readSFF",
	      sffFile, col, row, maxBases, nSample, randomSeed,
          PACKAGE="torrentR"
        )
	val
}
