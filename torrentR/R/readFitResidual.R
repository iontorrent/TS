readFitResidual <- function(beadParamFile,col,row,flow, minCol,maxCol,minRow,maxRow,minFlow,maxFlow,returnResErr,regXSize,regYSize,retRegStats,retRegFlowStats){

  val <- .Call("readFitResidualR",
               beadParamFile,col,row,flow,
               minCol,maxCol,
               minRow,maxRow,
               minFlow,maxFlow,
               returnResErr,
               regXSize,regYSize,
               retRegStats,retRegFlowStats,
               PACKAGE = "torrentR")

  val
 }


readFitResidualV2 <- function(beadParamFile,col,row,flow, minCol,maxCol,minRow,maxRow,minFlow,maxFlow,returnResErr,regXSize,regYSize,retRegStats,retRegFlowStats){

  val <- .Call("readFitResidualRV2",
               beadParamFile,col,row,flow,
               minCol,maxCol,
               minRow,maxRow,
               minFlow,maxFlow,
               returnResErr,
               regXSize,regYSize,
               retRegStats,retRegFlowStats,
               PACKAGE = "torrentR")

  val
 }
