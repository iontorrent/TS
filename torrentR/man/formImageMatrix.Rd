\name{formImageMatrix}
\alias{formImageMatrix}
\title{
  Transform (x,y,z) triples into a matrix suitable for supplying to image() or imageWithHist()
}
\description{
  Given a set of (x,y,z) triples along with the size of the x and y
  dimensions to which the data belong, returns a matrix of z values
  with number of columns equal to the x dimension and number of rows equal
  to the y dimension.

  Matrix entries are NA where there is no corresponding (x,y) value.  The
  matrix can be supplied to \code{\link{image}} or \code{\link{imageWithHist}}
  for plotting.
}
\usage{
  formImageMatrix(
    x,
    y,
    z,
    maxX,
    maxY
  )
}
\arguments{
  \item{x,y,z}{
    Vectors (of equal length) specifying the (x,y,z) triples.

    IMPORTANT: x and y values are expected to be 0-based.
  }
  \item{maxX,maxY}{
    The number of columns (maxX) and rows (maxY) in the returned matrix.
    All x values are expected to be strictly less than maxX and all y
    values strictly less than maxY.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{imageWithHist}}, \code{\link{bin2D}}, \code{\link{chipPlot}}
}
\examples{
nRow <- 20
nCol <- 50
x <- rep(0:(nCol-1),rep(nRow,nCol))
y <- rep(0:(nRow-1),nCol)
z <- formImageMatrix(x,y,rnorm(nCol*nRow,mean=x+y),maxX=nCol,maxY=nRow)
imageWithHist(z,header="Example",zlim=range(z))
}
