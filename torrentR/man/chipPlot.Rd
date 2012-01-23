\name{chipPlot}
\alias{chipPlot}
\title{
  Create a heatmap of data binned by (x,y) coordinates.
}
\description{
  Given a set of (x,y,z) triples, makes a heatmap of their values after doing
  some binning by (x,y) values.  Useful for making heatmaps of data by
  position on a chip, for example.
}
\usage{
  chipPlot(
    zCol,
    zRow,
    zVal,
    minCol=0,
    minRow=0,
    maxCol=NA,
    maxRow=NA,
    zlim=NA,
    nColBin=100,
    nRowBin=100,
    minBin=5,
    header="",
    histLim=NA,
    doPlot=TRUE,
    doInterpolate=FALSE,
    cex.header=1,
    summaryFunction=c("mean","median"),
    color=rgb(rep(0,256),seq(0,1,length=256),seq(1,0,length=256))
  )
}
\arguments{
  \item{zCol,zRow,zVal}{
    Vectors (of equal length) specifying the (x,y,z) triples.
    IMPORTANT: x and y values are expected to be 0-based.
  }
  \item{minCol,minRow}{
    The minimum x and y values to use when binning - zero by default.
    All x values should be greater than or equal to minCol, similar for y.
  }
  \item{maxCol,maxRow}{
    The maximum x and y values to use when binning.  If NA (the default) then
    the maximum will be set to 1 plus the max observed value.  All x values
    should be stricly less than maxCol, similar for maxRow.
  }
  \item{zlim}{
    A numeric vector of length 2 specifying lower and upper limits to
    which data will be truncated prior to plotting.  The default (if zlim
    is set to NA) is to truncate to the 2nd and 98th percentiles.
  }
  \item{nColBin,nRowBin}{
    The number of bins into which to aggregate data along X and Y axes,
    default is 100.
  }
  \item{minBin}{
    The minimum number of values per bin (default is 5).  Any bin having
    fewer than this number of values will have its entry set to NA.
  }
  \item{header}{
    Text string for plot title.  Default is no plot title.
  }
  \item{histLim}{
    A numeric vector of length 2 specifying lower and upper limits for the
    data going into the histogram plot.  If set to NA the default behaviour is
    to restrict to the range of the input image data, after applying any
    truncation implied by the zlim parameter.
  }
  \item{doPlot}{
    If set to FALSE no plot is generated, only effect is to generate
    return data.
  }
  \item{doInterpolate}{
    If set to TRUE, the binned and summarized values are bi-linearly
   interpolated back out chip-wide and returned.
  }
  \item{cex.header}{
    String expansion applied to plot title identified by header parameter.
  }
  \item{summaryFunction}{
    The function to use to summarize the data in each bin before plotting.
    Must be either "mean" (the default) or "median".
  }
  \item{color}{
    The color palette used for the image plot.  Default is a 256-step transition from blue to green.
  }
}
\value{
  \item{nColBin,nRowBin}{
    The number of column and row bins in the binned data.
  }
  \item{binnedCol,binnedRow,binnedVal}{
    The x,y and summarized aggregated z values for the binned data.
  }
  \item{sig}{
    If doInterpolate is set to TRUE, interpVal will contain z values
    after bi-linear interpolation back out to the dimensions of the input data.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{imageWithHist}}, \code{\link{formImageMatrix}}, \code{\link{bin2D}}
}
\examples{
nCol <- 200
nRow <- 100
x <- rep(0:(nCol-1),rep(nRow,nCol))
y <- rep(0:(nRow-1),nCol)
z <- rnorm(nCol*nRow,mean=x+y)
ret <- chipPlot(x,y,z,nColBin=20,nRowBin=10)
}
