
writeDat <- function(
  datFile,
  col=500,
  row=500,
  width=100,
  height=100,
  signal
) {

	val <- .Call("writeDat",
	      datFile, col, row, width, height, signal, 
          PACKAGE="torrentR"
        )
        return(val)
}
