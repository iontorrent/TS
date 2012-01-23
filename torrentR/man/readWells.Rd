\name{readWells}
\alias{readWells}
\title{
  Read signal data from a .wells file
}
\description{
  Reads signal data from a .wells file, offering various options to load specific subsets of information
  so that a manageable-sized chunk can be loaded into memory.  By default this function will try to read
  all flows of all wells, however in many cases that would involve reading unmanageable amounts of data
  into memory.  As a result it is likely that the most common use of this function will involve one or
  more of the flow,col,row,colMin,rowMin,colMax,rowMax options.
}
\usage{
  readWells(
    wellPath,
    col=NA,
    row=NA,
    bfMaskFile=NA,
    bfMaskFileName="bfmask.bin",
    colMin=NA,
    rowMin=NA,
    colMax=NA,
    rowMax=NA,
    ignoreBfMaskFile=FALSE,
    nCol=NA,
    nRow=NA,
    flow=numeric()
  )
}
\arguments{
  \item{wellPath}{
    Wells File to read.
  } 
  \item{flow}{
    Integer 1-based vector to enable loading signal data for only a specific subset of flows.  If this
    vector has zero-length (the default) then all available flows will be loaded.
  }
  \item{col,row}{
    Used to load signal data for only a specific subset of wells.  col and row are a pair
    of 0-based integer vectors specifying the (col,row) coordinates of the wells to load.  If these
    options are both set to NA then the set of wells to load is determined by the values of the
    colMin,colMax,rowMin,rowMax parameters.
  }
  \item{bfMaskFile}{
    The name of a bead find mask file from which to load up bead find-related information for the wells
    being read.  If this value is NA (the default) and if ignoreBfMaskFile is TRUE (the default) then
    the directory in which the wells file is located will be searched for a file named bfMaskFileName
    to load for beadFindMask information.
  }
  \item{bfMaskFileName}{
    The name of the bead find mask file to look for in the directory containing the wells file, for
    cases where the caller isn't explicitly specifying the bead find mask filename.
  }
  \item{colMin,colMax,rowMin,rowMax}{
    Used to load a rectangular sub-region of the chip.  These values are ignored if an explicit set
    of wells to load was specified by the (col,row) arguments.  All 4 values are 0-based.  If the min values
    are NA then they are replaced by 0 and if the max values are NA they are replaced by the largest possible
    coordinate.  So if all four of these options are at their default NA values the entire bead find mask
    will be read.
  }
  \item{ignoreBfMaskFile}{
    By default this is set to FALSE, but if it is TRUE then there will be no attempt to load up
    bead find mask information.  In that event, the return value will contain no bead find mask
    information.  Note that the bead find mask file is where the information on the number of 
    rows and columns is obtained (since they are not present in the wells file), so if ignoreBfMaskFile
    is set to true then nCol and nRow must be set.
  }
  \item{nCol,nRow}{
    If no bead find information is loaded then the number of columns and rows in the chip must be
    explicitly provided in this option.
  }
}
\value{
  The return value is a list with elements described below:
  \item{wellFile}{
    The name of the wells file from which data is loaded.
  }
  \item{nCol,nRow,nFlow}{
    The total number of columns, rows and flows represented in the wells file.
  }
  \item{nLoaded}{
    The total number of wells that have been read into memory.
  }
  \item{flow}{
    An integer vector (1-based) specifing the flows that were read into memory.
  }
  \item{flowBase}{
    A character vector specifying the nucleotide flowed in each one of the flows that were loaded.
  }
  \item{flowOrder}{
    A character string specifying the nucleotides flowed in the complete set of flows represented in the wells file
    (regardless of which flows have actually been loaded into memory)
  }
  \item{col,row}{
    Integer vectors (0-based) specifying the column and row coordinates of the wells that were loaded into memory.
  }
  \item{rank}{
    A numeric vector containing the rank values from each of the wells that were loaded into memory.
  }
  \item{beadFindMaskFile}{
    The bead find mask file that was read when reading the wells file (empty if none was read).
  }
  \item{mask}{
    A list of boolean vectors describing the wells that were loaded into memory.  If no bead find mask
    file was read th list will be empty.  Otherwise the list will contain entries as named below, for
    detail on what each represents see \code{\link{readBeadFindMask}}.
      empty
      bead
      live
      dud
      ambiguous
      tf
      lib
      pinned
      ignore
      washout
  }
  \item{signal}{
    A matrix with the raw signal data from the wells file.  There is a column for each flow and a row for each well.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readBeadFindMask}},
  \code{\link{readBeadFindMaskHeader}},
}
\examples{
# x1 <- readWells("/mydata/1.wells",flows=1:10)
# x2 <- readWells("/mydata/1.wells",col=0:4,row=0:4)
}
