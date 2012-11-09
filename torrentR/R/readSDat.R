readSDat <- function(
  datFile
) {
    val <- .Call("R_readSDat",
      datFile,
      PACKAGE="torrentR"
    )
  return(val)
}
