\name{bin2D}
\alias{bin2D}
\title{
  Aggregage (x,y,z) triples into bins on (x,y) dimensions.  Can be used
  to bin z data by (x,y) values for plotting as a heatmap at coarser
  resolution than the original data.
}
\description{
  Given (x,y,z) triples, returns a new set of (x,y,z) triples binned by (x,y)
}
\usage{
  bin2D(
    x,
    y,
    z,
    minX=0,
    minY=0,
    maxX=NA,
    maxY=NA,
    nBinX=100,
    nBinY=100,
    minBin=1
  )
}
\arguments{
  \item{x,y,z}{
    Vectors (of equal length) specifying the (x,y,z) triples.
  }
  \item{minX,minY}{
    The minimum x and y values to use when binning - zero by default.
    All x values should be greater than or equal to minX, similar for y.
  }
  \item{maxX,maxY}{
    The maximum x and y values to use when binning.  If NA (the default) then
    the maximum will be set to 1 plus the max observed value.  All x values
    should be stricly less than maxX, similar for maxY.
  }
  \item{nBinX,nBinY}{
    The number of (equal-width) bins into which to split the x and y data.
  }
  \item{minBin}{
    The minum number of observations required for each bin. If the actual
    number of data is less than this minum the bin in question is set to NA.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{imageWithHist}}, \code{\link{formImageMatrix}}, \code{\link{chipPlot}}
}
\examples{
# Example of a heatmap
nRow <- 20
nCol <- 50
x <- rep(0:(nCol-1),rep(nRow,nCol))
y <- rep(0:(nRow-1),nCol)
z <- rnorm(nCol*nRow,mean=x+y)
zMat <- formImageMatrix(x,y,z,maxX=nCol,maxY=nRow)
imageWithHist(zMat,header="Fine")

# Make it coarser
nRowCoarse <- nRow/5
nColCoarse <- nCol/5
coarse <- bin2D(x,y,z,nBinX=nColCoarse,nBinY=nRowCoarse)
zSummary <- unlist(lapply(z,mean))
zMatCoarse <- formImageMatrix(coarse$x,coarse$y,zSummary,maxX=nColCoarse,maxY=nRowCoarse)
imageWithHist(zMatCoarse,header="Coarse")
}
