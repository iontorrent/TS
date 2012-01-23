\name{readBeadFindMask}
\alias{readBeadFindMask}
\title{
  Read contents of a bead find mask file
}
\description{
  Reads the contents of a bead find mask file.  By default the entire bead find mask is read but there are options
  to enable loading either a rectangular sub-region or an arbitrary set of coordinates.
}
\usage{
  readBeadFindMask(
    beadFindFile,
    col=NA,
    row=NA,
    colMin=NA,
    rowMin=NA,
    colMax=NA,
    rowMax=NA
  )
}
\arguments{
  \item{beadFindFile}{
    Name of the bead find mask file to load.
  }
  \item{col,row}{
    Used to load the beadFind information for only a specific subset of wells.  col and row are a pair
    of 0-based integer vectors specifying the (col,row) coordinates of the wells to load.  If these
    options are both set to NA then the set of wells to load is determined by the values of the
    colMin,colMax,rowMin,rowMax parameters.
  }
  \item{colMin,colMax,rowMin,rowMax}{
    Used to load a rectangular sub-region of the chip.  These values are ignored if an explicit set
    of wells to load was specified by the (col,row) arguments.  All 4 values are 0-based.  If the min values
    are NA then they are replaced by 0 and if the max values are NA they are replaced by the largest possible
    coordinate.  So if all four of these options are at their default NA values the entire bead find mask
    will be read.
  }
}
\value{
  \item{beadFindMaskFile}{
    Name of the bead find mask file
  }
  \item{nCol,nRow}{
    Number of columns and rows in the associated chip
  }
  \item{col,row}{
    Integer vectors with the 0-based column and row coordinates of the wells for which bead find mask data was read.
  }
  \item{maskEmpty}{
    Boolean vector identifying the wells flagged as empty
  }
  \item{maskBead}{
    Boolean vector identifying the wells flagged as containing a bead
  }
  \item{maskLive}{
    Boolean vector identifying the wells flagged as live
  }
  \item{maskDud}{
    Boolean vector identifying the wells flagged as duds
  }
  \item{maskAmbiguous}{
    Boolean vector identifying the wells flagged as being ambiguous
  }
  \item{maskTF}{
    Boolean vector identifying the wells flagged as having a Test Fragment key signal
  }
  \item{maskLib}{
    Boolean vector identifying the wells flagged as having a library key signal
  }
  \item{maskPinned}{
    Boolean vector identifying the wells flagged as being pinned (i.e. saturated at some point in the run)
  }
  \item{maskIgnore}{
    Boolean vector identifying the wells flagged to be ignored in downstream analysis
  }
  \item{maskWashout}{
    Boolean vector identifying the wells flagged as having been washed out during the course of the run
  }
  \item{maskExclude}{
    Boolean vector identifying the wells flagged as being in the exclusion zone (not expected to yield any useful data)
  }
  \item{maskKeypass}{
    Boolean vector identifying the wells flagged as having passing key signal
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readBeadFindMaskHeader}},
  \code{\link{readWells}},
}
