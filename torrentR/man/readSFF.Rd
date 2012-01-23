\name{readSFF}
\alias{readSFF}
\title{
  Read SFF (Standard Flowgram File)
}
\description{
  Reads Ion Torrent SFF files, which contain base calls and flow signal values.
}
\usage{
  readSFF(
    sffFile,
    col=numeric(),
    row=numeric(),
    maxBases=150
  )
}
\arguments{
  \item{sffFile}{
    Name of the SFF file to load
  }
  \item{col,row}{
    As an alternative to returning the entire SFF file, an integer vector of 0-indexed col and row
    coordinates can be supplied to specify an arbitrary collection of wells.  If any of the wells
    that are specified are not found a warning will be issued and those that can be found are returned.
  }
  \item{maxBases}{
    The returned bases, quality scores and flowIndex values will be truncated to this length.  The
    identity of reads that have been truncated by this limit can be established be comparing the
    return values of length and fullLength which report the trimmed and untrimmed lengths respectively.
  }
}
\value{
  \item{nFlow}{
    The number of flows.
  }
  \item{col,row}{
    Vectors with the 0-indexed col and row coordinates of each read.
  }
  \item{length,fullLength}{
    Vectors with number of base calls in the possibly-truncated and full-length version of each read.
  }
  \item{clipQualLeft,clipQualRight,clipAdapterLeft,clipAdapterRight}{
    Vectors with the 1-indexed quality and adapter clip positions for each read.  See SFF format documentation for more info.
  }
  \item{flow}{
    Matrix of corrected flow signal values, one row for each read and one colum for each flow.
  }
  \item{base}{
    Vector of character strings with the called bases.  Each string is padded out with N to length maxBases as necessary.
  }
  \item{qual}{
    Matrix of per-base quality scores in Phred (-10log10) scale.  One row per read, one column per base position.
  }
  \item{flowIndex}{
    Matrix of flowIndex values mapping base calls to flows from which each base call originates.  Values are 1-indexed and
    each value is stored relative to the previous value in the array.
  }
}
\author{
  Simon Cawley
}
