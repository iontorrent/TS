\name{readIonBam}
\alias{readIonBam}
\title{
  Read Ion Torrent BAMs with flowspace data
}
\description{
  An Ion Torrent-optimized BAM reader
}
\usage{
  read(
    bamFile,
    col=numeric(),
    row=numeric(),
    maxBases=250,
    nSample=0,
    randomSeed=1,
    readGroups=character(),
    wantMappingData=TRUE,
    maxCigarLength=100
  )
}
\arguments{
  \item{bamFile}{
    Name of the BAM file to load
  }
  \item{col,row}{
    As an alternative to returning the entire BAM file, an integer vector of 0-indexed col and row
    coordinates can be supplied to specify an arbitrary collection of wells.  If any of the wells
    that are specified are not found a warning will be issued and those that can be found are returned.
    Read coordinates are determined by parsing read names of the form "HASH:row:col".
  }
  \item{maxBases}{
    The returned bases, quality scores and flowIndex values will be truncated to this length.  The
    identity of reads that have been truncated by this limit can be established be comparing the
    return values of length and fullLength which report the trimmed and untrimmed lengths respectively.
  }
  \item{nSample}{
    As an alternative to explicitly specifying a subset of reads to load via the
    col and row arguments, a number of reads to randomly sample can be specified
    via the nSample option.
  }
  \item{randomSeed}{
    Can be used to specify the random seed when getting a random sample of
    reads.  Must be a positive integer.
  }
  \item{readGroups}{
    A character vector specifying read groups to restrict to.  To see which
    read groups are availalbe, consider using readBamReadGroups()
  }
  \item{wantgMappingData}{
    If true, alignment fields will be parsed from the BAM file
  }
  \item{maxCigarLength}{
    Sets the maximum number of elements of the cigar string that will be returned.
  }
}
\value{
  \item{nFlow}{
    The number of flows.
  }
  \item{id}{
    The read name.
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
  \item{flowClipLeft,flowClipRight}{
    Vectors with the 0-indexed flows for the first aligned base and for the first base of any 3' adapter that was detected.
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
  \item{alignFlag}{
    The alignment flag
  }
  \item{alignBase}{
    Aligned query bases (padded with dashes)
  }
  \item{alignRefID}{
    ID of the reference sequenc to which the read aligns
  }
  \item{alignPos}{
    1-based leftmost alignment position
  }
  \item{alignMapq}{
    Mapping quality
  }
  \item{alignBin}{
    Bin associated with the alignment
  }
  \item{alignCigarType}{
    Character vector, each entry is the set of cigar operation types (MIDNSHPIX=).  Vector max
    length is determined by the maxCigarLength option.
  }
  \item{alignCigarLen}{
    Integer matrix with a row for each alignment, each row describes the length of the cigar operations for the
    corresponding read.  Number of columns is limited by the maxCigarLength option
  }
  \item{header}{
    The header section of the bam file as returned by readBamHeader()
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readBeadHeader}}
}
