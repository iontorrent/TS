\name{readDat}
\alias{readDat}
\title{
  Read raw Ion Torrent .dat files
}
\description{
  Reads Ion Torrent raw dat files, where one dat file corresponds to a single nucleotide flow.
  Can be used to read subsets of dat files by restricting to certain frames, wells and flows.
}
\usage{
  readDat(
    datFile,
    col=numeric(),
    row=numeric(),
    minCol=-1,
    maxCol=-1,
    minRow=-1,
    maxRow=-1,
    returnSignal=TRUE,
    returnWellMean=FALSE,
    returnWellSD=FALSE,
    returnWellLag=FALSE,
    uncompress=TRUE,
    doNormalize=FALSE,
    normStart=5,
    normEnd=20,
    XTCorrect=TRUE,
    baselineMinTime=0,
    baselineMaxTime=0.7,
    loadMinTime=0,
    loadMaxTime=-1
  )
}
\arguments{
  \item{datFile}{
    Character vector with names of the dat files to load
  }
  \item{col,row}{
    As an alternative to specifying a rectangular region, an integer vector of 0-indexed col and row
    coordinates can be supplied to specify an arbitrary collection of wells.  When using this approach
    the minimum spanning rectagle of all requested wells will be read from disk.  So when the wells
    are spatially confined the read will be fast, when they are dispersed it will be slower.
  }
  \item{minCol,maxCol,minRow,maxRow}{
    Can be used to specify one or more rectangular sub-regions.  Values are 0-indexed for min and 1-indexed for max,
    so the values for max should be strictly greater than the values for min.  For multiple regions, set each to
    a vector whose length is the number of regions sought.  Setting maxCol or maxRow to -1
    leads to their being re-set to the maximum possible value.  Default is one whole-chip region.
  }
  \item{returnSignal}{
    Specifies whether or not to return a matrix of raw signal data.
  }
  \item{returnWellMean,returnWellSD,returnWellLag}{
    Specifies whether or not to return a matrix of per-well mean, SD and SD of lag-1 differences for the signal data that would be returned.
  }
  \item{uncompress}{
    Specifies whether or not to uncompress if dat is written with Variable Framerate Compression (VFC)
  }
  \item{doNormalize,normStart,normEnd}{
    Specifies if normalization should be applied - this is a subtraction from each well of the average
    value in the frame range [normStart,normEnd].  Off by default, but will be forced on if XTCorrect
    is on.
  }
  \item{XTCorrect}{
    Apply electrical cross-talk correct to undo an electrical cross-talk that occurs on the 316 and 318
    chips.  Signal between pixels in columns that are multiples of 4 apart is convolved and needs to be
    deconvolved.  Not used for 314.  If XTCorrect is true (the default) then doNormalize will be forced.
  }
  \item{baselineMinTime, baselineMaxTime}{
    Controls whether or not reads should be baselined.  If baselineMaxTime is greater than baselineMinTime
    then each well will have the average of the flows completely contained within the timeframe subtracted.
    The average is weighted by the duration spanned by each frame used.
  }
  \item{loadMinTime,loadMaxTime}{
    Specify times, in seconds, to determine which frames are returned.
    If loadMaxTime is less than loadMinTime then all frames are returned.
  }
}
\value{
  \item{datFile}{
    Character vector with the names of the dat files loaded.
  }
  \item{nCol,nRow,nFrame,nFlow}{
    The number of columns, rows, frames and flows represented in the full dat file.
    If a subset of wells or frames has been requested nCol, nRow and nFrame will still
    describe the full dat file.  nFrame will describe the number of compressed frames
    in a dat file written with Variable Framerate Compression (VFC), unless uncompress
    is set to true in which case nFrame will describe the number of uncompressed frames.
  }
  \item{col,row}{
    Integer vectors with the 0-indexed coordinates of the wells returned.
  }
  \item{frameStart,frameEnd}{
    Vector specifying the frame start/end time in seconds.  If multiple dat files are read this vector repeats such that its length is equal to the total number of frames loaded across all flows.
  }
  \item{signal}{
    Numeric matrix of the raw signal data.  This is only returned if returnSignal is TRUE.  One column per frame loaded and one row per well loaded.  In the case of multiple dats the number of columns is equal to the number of dats times the number of frames loaded per dat.
  }
  \item{wellMean,wellSD}{
    Numeric matrices of the mean and SD of the signal data that would be returned.  Only returned if returnWellMean
    and returnWellSD are true, respectively.  One row per well and one column per dat.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readDatCollection}},
}
