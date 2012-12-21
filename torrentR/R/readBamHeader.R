readBamHeader <- function(
  bamFile
) {

	if(!file.exists(bamFile))
		stop(sprintf("BAM file not found: %s",bamFile))

	header    <- .Call("readBamHeader",    bamFile, PACKAGE="torrentR")
	readGroup <- .Call("readBamReadGroup", bamFile, PACKAGE="torrentR")
	sequence  <- .Call("readBamSequence",  bamFile, PACKAGE="torrentR")
	return(list(ReadGroup=readGroup, Sequence=sequence))
}
