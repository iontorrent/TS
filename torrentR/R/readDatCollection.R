readDatCollection <- function(
  datDir=NA,
  analysisDir=NA,
  minFlow=1,
  maxFlow=-1,
  col=numeric(),
  row=numeric(),
  minCol=0,
  maxCol=-1,
  minRow=0,
  maxRow=-1,
  returnSignal=TRUE,
  returnWellMean=FALSE,
  returnWellSD=FALSE,
  uncompress=TRUE,
  baselineMinTime=0,
  baselineMaxTime=0.7,
  loadMinTime=0,
  loadMaxTime=-1
) {

  if(is.na(datDir) & is.na(analysisDir))
    stop("Must specify either datDir or analysisDir\n");
  if(!is.na(datDir) & !is.na(analysisDir))
    stop("Do not specify both datDir and analysisDir, just one please\n");

  if(is.na(datDir)) {
    datDir  <- findDatDir(analysisDir)
    if(is.null(datDir))
      stop(sprintf("Unable to locate dat directory for %s\n",analysisDir))
  }

  dat <- readDatList(datDir)
  if(length(dat$datFile)==0)
    stop(sprintf("Unable to locate any dat images in directory %s\n",datDir))
  if(maxFlow < 1)
    maxFlow <- max(dat$datFlows)
  if(maxFlow > max(dat$datFlows)) {
    warning(sprintf("%d flows requested but only %d available, limiting to the available flows\n",maxFlow,max(dat$datFlows)))
    maxFlow <- max(dat$datFlows)
  }
  if(minFlow > maxFlow)
    stop("minFlow should be less than or equal to maxFlow\n")
  flowRange <- minFlow:maxFlow

  if(!all(is.element(flowRange,dat$datFlows)))
    stop(sprintf("Missing at least one dat file for the requested flow range %d:%d\n",minFlow,maxFlow))
  names(dat$datFiles) <- as.character(dat$datFlows)

  datFiles <- sprintf("%s%s%s",datDir,.Platform$file.sep,dat$datFiles[as.character(flowRange)])
  res <- readDat(
    datFiles,
    col=col,
    row=row,
    minCol=minCol,
    maxCol=maxCol,
    minRow=minRow,
    maxRow=maxRow,
    returnSignal=returnSignal,
    returnWellMean=returnWellMean,
    returnWellSD=returnWellSD,
    uncompress=uncompress,
    baselineMinTime=baselineMinTime,
    baselineMaxTime=baselineMaxTime,
    loadMinTime=loadMinTime,
    loadMaxTime=loadMaxTime
  )
  res$flow <- flowRange

  return(res)
}

