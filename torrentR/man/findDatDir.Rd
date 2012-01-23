\name{findDatDir}
\alias{findDatDir}
\title{
  Finds the location of the directory of dat files for a given analysis run.
}
\description{
  Given the path to a directory of analysis results, this function returns the path to the associated directory of raw DAT files.
  The mechanism is somewhat fragile - the location is parsed out of the processParameters.txt file that is expected to be located
  in the analysis directory.  For "--from-wells" analysis runs the processParameters.txt unfortunately doesn't include the information
  about the location of the raw DATs, nor in fact does any other file in the analysis directory.
}
\usage{
  findDatDir(analysisDir,paramFile="processParameters.txt",paramName="dataDirectory")
}
\arguments{
  \item{analysisDir}{
    The path to the directory of analysis results.
  }
  \item{paramFile}{
    The name of the file in the analysis directory that is expected to contain run information - "processParameters.txt" by default.
  }
  \item{paramName}{
    The prefix for the line that is expected to provide the information - "dataDirectory" by default.
  }
}
\value{
  The location of the raw results directory is returned, or NULL if it couldn't be found.
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readDatList}},
}
