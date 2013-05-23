#return the stack of reads relevant to a single variant position
#and possibly some additional data

readIonBamStack <- function(
  bamFile,
  variant_contig,
  variant_position
) {

	if(!file.exists(bamFile))
		stop(sprintf("BAM file not found: %s",bamFile))

	val <- .Call("readBamStackAtPosition",
	      bamFile, variant_contig, variant_position,
          PACKAGE="torrentR"
        )
	return(val)
}

