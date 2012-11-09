readBeadParam <- function(beadParamFile,minCol,maxCol,minRow,maxRow,minFlow,maxFlow){
  val <- .Call("readBeadParamR",
               beadParamFile,
               minCol,maxCol,
               minRow,maxRow,
               minFlow,maxFlow,
               PACKAGE = "torrentR")
  val
 }


readBeadParamV2 <- function(beadParamFile,minCol,maxCol,minRow,maxRow,minFlow,maxFlow){
  val <- .Call("readBeadParamRV2",
               beadParamFile,
               minCol,maxCol,
               minRow,maxRow,
               minFlow,maxFlow,
               PACKAGE = "torrentR")
  val
 }
