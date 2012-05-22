readWells <- function(
	wellPath,
	col=NA,
	row=NA,
	bfMaskFile=NA,
	bfMaskFileName="analysis.bfmask.bin",
	colMin=NA,
	rowMin=NA,
	colMax=NA,
	rowMax=NA,
	ignoreBfMaskFile=FALSE,
	nCol=NA,
	nRow=NA,
	flow=numeric()
) {
	wellDir  <- dirname(wellPath)
	wellFile <- basename(wellPath)

  #we've swapped files around

	# Check for a bfMaskFile unless specifically asked not to via ignoreBfMaskFile=TRUE
	if(is.na(bfMaskFile) & !ignoreBfMaskFile) {

	    bfMaskFile <- paste(wellDir,.Platform$file.sep,bfMaskFileName,sep="")
	    if(file.exists(bfMaskFile)) {
		    if(file.access(bfMaskFile,mode=4)!=0) {
		    warning(paste("file ",bfMaskFile," exists but is not readable, skipping\n",sep=""))
		    bfMaskFile <- NA
		}
	    } else {
                warning(paste("did not find bfmask file ",bfMaskFile,", proceeding without it\n",sep=""))
		     bfMaskFile <- NA
	    }

	}


	# Read the bfMaskFile if we have one
	if(!is.na(bfMaskFile)) {
	    header <- readBeadFindMaskHeader(bfMaskFile)
	    nCol <- header$nCol
	    nRow <- header$nRow
	} else if(any(is.na(nCol)) | any(is.na(nRow))) {
	    stop(paste("Either nCol and nRow must be specified, or a ",bfMaskFileName," file must be available\n",sep=""))
	}
	colrow <- expandColRow(col,row,colMin,rowMin,colMax,rowMax,nCol,nRow)

	# Set up the return object
	val <- list()

	# Read the beadFindMask data if available, put into return object
	if(!is.na(bfMaskFile)) {
	  data.bf <- readBeadFindMask(bfMaskFile,colrow$col,colrow$row)
	    val$beadFindMaskFile <- data.bf$beadFindMaskFile
	    val$mask <- list(
		empty     = data.bf$maskEmpty,
		bead      = data.bf$maskBead,
		live      = data.bf$maskLive,
		dud       = data.bf$maskDud,
		ambiguous = data.bf$maskAmbiguous,
		tf        = data.bf$maskTF,
		lib       = data.bf$maskLib,
		pinned    = data.bf$maskPinned,
		ignore    = data.bf$maskIgnore,
		washout   = data.bf$maskWashout
	    )
	}

	# read data from 1.wells file and populate into return object
	data.wells <- .Call("readWells",
          wellDir,
          wellFile,
          nCol,
          nRow,
          colrow$col,
          colrow$row,
	  flow-1,
          PACKAGE="torrentR"
        )
	if(!all(is.finite(data.wells$signal)))
          warning(sprintf("Encountered some infinite signal values in wells file %s%s%s\n",wellDir,.Platform$file.sep,wellFile))

	if(length(flow)==0)
	  flow <- 1:data.wells$nFlow

	val$col          <- colrow$col
	val$row          <- colrow$row
	val$flow         <- flow
	val$flowBase     <- sapply(flow,function(x){substr(data.wells$flowOrder,x,x)})
	val$wellFile     <- wellPath
	val$nCol         <- nCol
	val$nRow         <- nRow
	val$nLoaded      <- length(colrow$col)
	val$nFlow        <- data.wells$nFlow
	val$flowOrder    <- substr(data.wells$flowOrder,1,data.wells$nFlow)
	val$rank         <- data.wells$rank
	val$signal       <- data.wells$signal

	return(val)
}
