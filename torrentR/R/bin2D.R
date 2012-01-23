bin2D <- function(x,y,z,minX=0,minY=0,maxX=NA,maxY=NA,nBinX=100,nBinY=100,minBin=1) {

  if(is.na(maxX))
    maxX <- max(x)+1
  if(is.na(maxY))
    maxY <- max(y)+1

  # Assign (x,y,z) triples to bins
  xBin <- floor(nBinX*(x-minX)/(maxX-minX))
  yBin <- floor(nBinY*(y-minY)/(maxY-minY))
  zBin <- yBin*nBinX+xBin

  # Aggregate to per-bin groups, dropping bins with too few entries
  zCollapse <- split(z,zBin)
  binSize <- unlist(lapply(zCollapse,length))
  if(all(binSize < minBin))
    stop("All bins have fewer than minBin values")
  zCollapse <- zCollapse[binSize >= minBin]
  indexCollapse <- as.numeric(names(zCollapse))
  xCollapse <- indexCollapse %% nBinX
  yCollapse <- floor(indexCollapse / nBinX)

  return(list(
    x = xCollapse,
    y = yCollapse,
    z = zCollapse
  ))
}
