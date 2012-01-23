\name{readDatList}
\alias{readDatList}
\title{
  Returns the list of all DAT files and associated flow numbers for a run.
}
\description{
  Given a directory which may contain raw DAT files for a run, returns the names of any DAT files and associated flow numbers.
  Searches for DAT files by finding files whose name matches the expected regular expression acq_####.dat
}
\usage{
  readDatList(datDir)
}
\arguments{
  \item{datDir}{
    The directory to search.
  }
}
\value{
  The return value is a list with two entries
  \item{datFiles}{
    A character vector specifying the dat file names.
  }
  \item{datFlows}{
    A numeric vector specifying the flow number for each dat file.  Numbering starts at 1, so acq_0000.dat would be flow 1, and so on.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{findDatDir}},
}
