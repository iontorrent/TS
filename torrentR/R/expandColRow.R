expandColRow<- function(col=NA,row=NA,colMin=NA,rowMin=NA,colMax=NA,rowMax=NA,nCol,nRow) {

	numNA <- any(is.na(col)) + any(is.na(row))
	if(numNA == 1) {
	    stop("specify either both col and row or neither")
	} else if(numNA == 2) {
	    if(is.na(colMin))
		colMin <- 0
	    if(is.na(rowMin))
		rowMin <- 0
	    if(is.na(colMax))
		colMax <- nCol-1
	    if(is.na(rowMax))
		rowMax <- nRow-1
	    temp <- expand.grid(colMin:colMax,rowMin:rowMax)
	    col <- temp[,1]
	    row <- temp[,2]
	}

	return(list(
	    col = col,
	    row = row
	))

}
