readSFF <- function(
  sffFile,
  col=numeric(),
  row=numeric(),
  maxBases=150
) {

	val <- .Call("readSFF",
	      sffFile, col, row, maxBases,
          PACKAGE="torrentR"
        )
	val
}
