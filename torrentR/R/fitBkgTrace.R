fitBkgTrace <- function(
  sig,
  nFrame,
  nFlow
) {

  val <- .Call("fitBkgTrace",
    sig, nFrame, nFlow,
    PACKAGE="torrentR"
  )
  val$bkg
}
