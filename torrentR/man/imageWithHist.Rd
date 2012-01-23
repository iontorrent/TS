\name{imageWithHist}
\alias{imageWithHist}
\title{
  Create a heatmap with an associated histogram.
}
\description{
  A wrapper around the image() function which additionally creates a histogram
  giving a sense of the distribution of the values forming the heatmap.
}
\usage{
  imageWithHist (
    z,
    zlim=NA,
    header="",
    histLim=NA,
    xaxt="n",
    yaxt="n",
    nHistBar=100,
    cex.header=1,
    col=rgb(rep(0,256),seq(0,1,length=256),seq(1,0,length=256)),
    ...
  )
}
\arguments{
  \item{z}{
    The data to be passed along to the image() function, modulo whatever
    truncation is applied due to the setting of the zlim parameter.
  }
  \item{zlim}{
    A numeric vector of length 2 specifying lower and upper limits to
    which data will be truncated prior to plotting.  The default (if zlim
    is set to NA) is to truncate to the 2nd and 98th percentiles.
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
  \item{xaxt,yaxt}{
    The axis type for the x and y axes of the image plot.  See par() for
    more info.
  }
  \item{nHistBar}{
    The number of bars in the histogram - default is 100.
  }
  \item{cex.header}{
    String expansion applied to plot title identified by header parameter.
  }
  \item{col}{
    The color palette used for the image plot.  Default is a 256-step transition from blue to green.
  }
  \item{...}{
    Any additional options get passed through to the call to image()
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{formImageMatrix}}, \code{\link{bin2D}}, \code{\link{chipPlot}}
}
\examples{
nRow <- 20
nCol <- 50
x <- rep(0:(nCol-1),rep(nRow,nCol))
y <- rep(0:(nRow-1),nCol)
z <- formImageMatrix(x,y,rnorm(nCol*nRow,mean=x+y),maxX=nCol,maxY=nRow)
imageWithHist(z,header="Example",zlim=range(z))
}
