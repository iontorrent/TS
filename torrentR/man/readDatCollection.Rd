\name{readDatCollection}
\alias{readDatCollection}
\title{
  Read groups of raw Ion Torrent .dat files
}
\description{
  Reads groups of Ion Torrent raw dat files, where one dat file corresponds to a single nucleotide flow.
  Can be used to read subsets of data by restricting to certain frames or wells.
}
\usage{
  readDatCollection(
    datDir=NA,
    analysisDir=NA,
    minFlow=1,
    maxFlow=-1,
    col=numeric(),
    row=numeric(),
    minCol=0,
    maxCol=-1,
    minRow=0,
    maxRow=-1,
    returnSignal=TRUE,
    returnWellMean=FALSE,
    returnWellSD=FALSE,
    uncompress=TRUE,
    baselineMinTime=0,
    baselineMaxTime=0.7,
    loadMinTime=0,
    loadMaxTime=-1
  )
}
\arguments{
  \item{datDir,analysisDir}{
    Exactly one of these must be specified.  If analysisDir is set then datDir, the location of the dat files, will be determined automatically.
    Otherwise the dat files are assumed to be located at datDir.
  }
  \item{minFlow,maxFlow}{
    Specify the first and last flows that should be loaded (1-indexed).  Setting maxFlow to -1 leads
    to it being re-set to the maximum available flow.
  }
  \item{col,row}{
    As an alternative to specifying a rectangular region, an integer vector of 0-indexed col and row
    coordinates can be supplied to specify an arbitrary collection of wells.
  }
  \item{minCol,maxCol,minRow,maxRow}{
    Can be used to specify a rectangular sub-region.  Values are 0-indexed.  Setting maxCol or maxRow to -1
    leads to their being re-set to the maximum possible value.
  }
  \item{baselineMinTime, baselineMaxTime}{
    Specify the time interval to be used for determination of well-specific baseline to subtract.
  }
  \item{returnSignal, returnWellMean, returnWellSD}{
    Specify what kind of data are returned.  If returnSignal is true the per-frame data are returned for
    all frames in the requested time interval.  If returnWellMean and returnWellSD the per-well mean and
    standard deviation of signal is returned.
  }
  \item{uncompress}{
    If true and if the data are stored with Variable Framerate Compression, the data are interpolated back out to
    all frames, at the cost of speed and memory.  If false, compressed data are returned as stored, which can be
    significantly faster and more compact.
  }
  \item{loadMinTime,loadMaxTime}{
    Specify the time interval for which to return data.  Setting loadMaxTime to -1 leads
    to it being re-set to the maximum possible value
  }
}
\value{
  The returned value is a list with the same entries as those returned by readDat (which this function wraps).
  The following additional entries are also set:
  \item{flow}{
    A vector specifying the flows that were loaded.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readDat}},
}
