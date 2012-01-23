readBeadFindMaskHeader <- function(beadFindFile) {

        if(!file.exists(beadFindFile))
          stop(paste("file ",beadFindFile," does not exist\n",sep=""))
        if(file.access(beadFindFile,mode=4)!=0)
          stop(paste("file ",beadFindFile," exists but is not readable\n",sep=""))

	val <- .Call("readBeadFindMaskHeader",
          beadFindFile,
          PACKAGE="torrentR"
        )
	val
}
