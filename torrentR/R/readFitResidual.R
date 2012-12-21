readFitResidual <- function(beadParamFile,
                              col = numeric(),
                              row = numeric(),
                              flow = numeric(),
                              minCol,maxCol,
                              minRow,maxRow,
                              minFlow,maxFlow,
                              returnResErr = FALSE,
                              regXSize = 100,
                              regYSize = 100,
                              retRegStats = TRUE,
                              retRegFlowStats = TRUE){

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

readFitResidualV2 <- function(beadParamFile,
                              col = numeric(),
                              row = numeric(),
                              flow = numeric(),
                              minCol,maxCol,
                              minRow,maxRow,
                              minFlow,maxFlow,
                              returnResErr = FALSE,
                              regXSize = 100,
                              regYSize = 100,
                              retRegStats = TRUE,
                              retRegFlowStats = TRUE){

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
