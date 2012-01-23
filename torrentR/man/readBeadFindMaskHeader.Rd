\name{readBeadFindMaskHeader}
\alias{readBeadFindMaskHeader}
\title{
  Read header section of a bead find mask file
}
\description{
  Read the header section of a bead find mask file (often named
  bfmask.bin) to determine the number of columns and rows in the
  corresponding chip.
}
\usage{
  readBeadFindMaskHeader(beadFindFile)
}
\arguments{
  \item{beadFindFile}{
    The bead find mask file to be read.
  }
}
\value{
  \item{nCol}{
    Number of columns in the associated chip
  }
  \item{nRow}{
    Number of rows in the associated chip
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readBeadFindMask}},
  \code{\link{readWells}},
}
\examples{
#bfHeader <- readBeadFindMaskHeader("/data/my_analysis_directory/bfmask.bin")
}
