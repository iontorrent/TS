readBeadFindMask <- function(beadFindFile,col=NA,row=NA,colMin=NA,rowMin=NA,colMax=NA,rowMax=NA) {

        if(!file.exists(beadFindFile))
          stop(paste("file ",beadFindFile," does not exist\n",sep=""))
        if(file.access(beadFindFile,mode=4)!=0)
          stop(paste("file ",beadFindFile," exists but is not readable\n",sep=""))

	header <- readBeadFindMaskHeader(beadFindFile)
	colrow <- expandColRow(col,row,colMin,rowMin,colMax,rowMax,header$nCol,header$nRow)

	val <- .Call("readBeadFindMask",
          beadFindFile,
          colrow$col,
          colrow$row,
          PACKAGE="torrentR"
        )
	val
}
